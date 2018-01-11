#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
"""
from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb

from chem_tensorflow import ChemModel
from utils import glorot_init, SMALL_NUMBER


class SparseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 100000,
            'use_edge_bias': False,
            'use_edge_msg_avg_aggregation': True,
            'tie_gnn_layers': False,
            'graph_rnn_cell': 'GRU',  # GRU or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int64, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int64, [None, 2], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['edge_weights'] = []
        self.weights['edge_biases'] = []
        self.weights['rnn_cells'] = []
        for step in range(self.params['num_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % step):
                if step == 0 or not(self.params['tie_gnn_layers']):
                    self.weights['edge_weights'].append(tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                                                    name='gnn_edge_weights_%i' % step))

                    if self.params['use_edge_bias']:
                        self.weights['edge_biases'].append(tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                                                       name='gnn_edge_biases_%i' % step))

                    cell_type = self.params['graph_rnn_cell'].lower()
                    if cell_type == 'gru':
                        cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                    elif cell_type == 'rnn':
                        cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                    else:
                        raise Exception("Unknown RNN cell type '%s'." % cell_type)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                         state_keep_prob=self.placeholders['graph_state_keep_prob'])
                    self.weights['rnn_cells'].append(cell)

    def compute_final_node_representations(self) -> tf.Tensor:
        with tf.variable_scope('gnn_scope'):
            cur_node_states = self.placeholders['initial_node_representation']  # number of nodes in batch v x D
            num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int64)[0]

            adjacency_matrices = []  # type: List[tf.SparseTensor]
            for adjacency_list_for_edge_type in self.placeholders['adjacency_lists']:
                # adjacency_list_for_edge_type (shape [-1, 2]) includes all edges of type e_type of a sparse graph with v nodes (ids from 0 to v).
                adjacency_matrix_for_edge_type = tf.SparseTensor(indices=adjacency_list_for_edge_type,
                                                                 values=tf.ones_like(
                                                                     adjacency_list_for_edge_type[:, 1],
                                                                     dtype=tf.float32),
                                                                 dense_shape=[num_nodes, num_nodes])
                adjacency_matrices.append(adjacency_matrix_for_edge_type)

            for step in range(self.params['num_timesteps']):
                effective_step = 0 if self.params['tie_gnn_layers'] else step
                with tf.variable_scope('gnn_layer_%i' % effective_step):
                    incoming_messages = []  # list of v x D

                    # Collect incoming messages per edge type
                    for adjacency_matrix in adjacency_matrices:
                        incoming_messages_per_type = tf.sparse_tensor_dense_matmul(adjacency_matrix,
                                                                                   cur_node_states)  # v x D
                        incoming_messages.extend([incoming_messages_per_type])

                    # Pass incoming messages through linear layer:
                    incoming_messages = tf.concat(incoming_messages, axis=1)  # v x [2 *] edge_types
                    messages_passed = tf.matmul(incoming_messages,
                                                self.weights['edge_weights'][effective_step])  # v x D

                    if self.params['use_edge_bias']:
                        messages_passed += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                     self.weights['edge_biases'][effective_step])  # v x D

                    if self.params['use_edge_msg_avg_aggregation']:
                        num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                           keep_dims=True, axis=-1)  # v x 1
                        messages_passed /= num_incoming_edges + SMALL_NUMBER

                    # pass updated vertex features into RNN cell
                    cur_node_states = self.weights['rnn_cells'][effective_step](messages_passed, cur_node_states)[1]  # v x D

            return cur_node_states

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]

        # Sum up all nodes per-graph
        num_nodes = tf.shape(gate_input, out_type=tf.int64)[0]
        graph_nodes = tf.SparseTensor(indices=self.placeholders['graph_nodes_list'],
                                      values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0],
                                                          dtype=tf.float32),
                                      dense_shape=[self.placeholders['num_graphs'], num_nodes])  # [g x v]
        return tf.squeeze(tf.sparse_tensor_dense_matmul(graph_nodes, gated_outputs), axis=[-1])  # [g]

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        processed_graphs = []
        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"],
                                     "labels": [d["targets"][task_id][0] for task_id in self.params['task_ids']]})

        if is_training_data:
            np.random.shuffle(processed_graphs)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['labels'][task_id] = None

        return processed_graphs

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0
            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = self.num_edge_types + edge_type
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.extend(
                    (num_graphs_in_batch, node_offset + i) for i in range(num_nodes_in_graph))
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                target_task_values = []
                target_task_mask = []
                for target_val in cur_graph['labels']:
                    if target_val is None:  # This is one of the examples we didn't sample...
                        target_task_values.append(0.)
                        target_task_mask.append(0.)
                    else:
                        target_task_values.append(target_val)
                        target_task_mask.append(1.)
                batch_target_task_values.append(target_task_values)
                batch_target_task_mask.append(target_task_mask)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                self.placeholders['graph_nodes_list']: np.array(batch_graph_nodes_list, dtype=np.int32),
                self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1,0]),
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict


def main():
    args = docopt(__doc__)
    try:
        model = SparseGGNNChemModel(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()