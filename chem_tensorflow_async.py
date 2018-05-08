#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_async.py [options]

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


def bfs_visit(outgoing_edges: Dict[int, Sequence[int]], node_depths: Dict[int, int], v: int, depth: int):
    # Already seen, skip:
    if v in node_depths:
        return
    node_depths[v] = depth
    for (_, __, w) in outgoing_edges[v]:
        bfs_visit(outgoing_edges, node_depths, w, depth + 1)


class AsyncGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'num_nodes': 100000,
            'use_edge_bias': False,

            'propagation_rounds': 4,  # Has to be an even number
            'propagation_substeps': 15,

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

        # Initial nodes I_{r}: Node IDs that will have no incoming edges in round r.
        self.placeholders['initial_nodes'] = [tf.placeholder(tf.int32, [None], name="initial_nodes_round%i" % prop_round)
                                              for prop_round in range(self.params['propagation_rounds'])]

        # Sending nodes S_{r,s,e}: Source node ids of edges propagating in step s of round r.
        # Restrictions: If v in S_{r,s,e}, then v in R_{r,s'} for s' < s or v in I_{r}
        self.placeholders['sending_nodes'] = [[[tf.placeholder(tf.int32,
                                                               [None],
                                                               name="sending_nodes_round%i_step%i_edgetyp%i" % (prop_round, step, edge_typ))
                                                for edge_typ in range(self.num_edge_types)]
                                               for step in range(self.params['propagation_substeps'])]
                                              for prop_round in range(self.params['propagation_rounds'])]

        # Normalised edge target nodes T_{r,s}: Targets of edges propagating in step s of round r, normalised to a
        # continuous range starting from 0. This is used for aggregating messages from the sending nodes.
        self.placeholders['msg_targets'] = [[tf.placeholder(tf.int32,
                                                            [None],
                                                            name="msg_targets_nodes_round%i_step%i" % (prop_round, step))
                                             for step in range(self.params['propagation_substeps'])]
                                            for prop_round in range(self.params['propagation_rounds'])]


        # Receiving nodes R_{r,s}: Target node ids of aggregated messages in propagation step s of round r.
        # Restrictions: If v in R_{r,s}, v not in R_{r,s'} for all s' != s and v not in I_{r}
        self.placeholders['receiving_nodes'] = [[tf.placeholder(tf.int32,
                                                                [None],
                                                                name="receiving_nodes_round%i_step%i" % (prop_round, step))
                                                 for step in range(self.params['propagation_substeps'])]
                                                for prop_round in range(self.params['propagation_rounds'])]

        # Number of receiving nodes N_{r,s}
        # Restrictions: N_{r,s} = len(R_{r,s})
        self.placeholders['receiving_node_num'] = [tf.placeholder(tf.int32,
                                                                  [self.params['propagation_substeps']],
                                                                  name="receiving_nodes_num_round%i" % (prop_round,))
                                                   for prop_round in range(self.params['propagation_rounds'])]

        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['edge_weights'] = [tf.Variable(glorot_init([h_dim, h_dim]), name='gnn_edge_weights_typ%i' % e_typ)
                                        for e_typ in range(self.num_edge_types)]

        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = [tf.Variable(np.zeros([h_dim], dtype=np.float32), name='gnn_edge_biases_typ%i' % e_typ)
                                           for e_typ in range(self.num_edge_types)]

        cell_type = self.params['graph_rnn_cell'].lower()
        if cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             state_keep_prob=self.placeholders['graph_state_keep_prob'])
        self.weights['rnn_cells'] = cell

    def compute_final_node_representations(self) -> tf.Tensor:
        cur_node_states = self.placeholders['initial_node_representation']

        for prop_round in range(self.params['propagation_rounds']):
            with tf.variable_scope('prop_round%i' % (prop_round,)):
                # ---- Declare and fill tensor arrays used in tf.while_loop:
                sending_nodes_ta = tf.TensorArray(tf.int32,
                                                  infer_shape=False,
                                                  element_shape=[None],
                                                  size=self.params['propagation_substeps'] * self.num_edge_types,
                                                  name='sending_nodes')
                msg_targets_ta = tf.TensorArray(tf.int32,
                                                infer_shape=False,
                                                element_shape=[None],
                                                size=self.params['propagation_substeps'],
                                                name='msg_targets')
                receiving_nodes_ta = tf.TensorArray(tf.int32,
                                                    infer_shape=False,
                                                    element_shape=[None],
                                                    size=self.params['propagation_substeps'],
                                                    clear_after_read=False,
                                                    name='receiving_nodes')
                receiving_node_num_ta = tf.TensorArray(tf.int32,
                                                       infer_shape=False,
                                                       element_shape=[],
                                                       size=self.params['propagation_substeps'],
                                                       name='receiving_nodes_num')

                for step in range(self.params['propagation_substeps']):
                    for edge_typ in range(self.num_edge_types):
                        sending_nodes_ta = sending_nodes_ta.write(step * self.num_edge_types + edge_typ,
                                                                  self.placeholders['sending_nodes'][prop_round][step][edge_typ])
                    msg_targets_ta = msg_targets_ta.write(step, self.placeholders['msg_targets'][prop_round][step])
                    receiving_nodes_ta = receiving_nodes_ta.write(step, self.placeholders['receiving_nodes'][prop_round][step])
                receiving_node_num_ta = receiving_node_num_ta.unstack(self.placeholders['receiving_node_num'][prop_round])

                new_node_states_ta = tf.TensorArray(tf.float32,
                                                    infer_shape=False,
                                                    element_shape=[self.params['hidden_size']],
                                                    size=tf.shape(cur_node_states)[0],
                                                    clear_after_read=False,
                                                    name='new_node_states')

                # ---- Actual propagation schedule implementation:
                # Initialize the initial nodes with their state from last round:
                new_node_states_ta = new_node_states_ta.scatter(self.placeholders['initial_nodes'][prop_round],
                                                                tf.gather(cur_node_states, self.placeholders['initial_nodes'][prop_round]))

                def do_substep(substep_id, new_node_states_ta):
                    # For each edge active in this substep, pull source state and transform:
                    sent_messages = []
                    for edge_typ in range(self.num_edge_types):
                        sending_states = new_node_states_ta.gather(sending_nodes_ta.read(substep_id * self.num_edge_types + edge_typ))
                        messages = tf.matmul(sending_states, self.weights['edge_weights'][edge_typ])
                        if self.params['use_edge_bias']:
                            messages += self.weights['edge_biases'][edge_typ]
                        sent_messages.append(messages)

                    # Stack all edge messages and aggregate as sum for each receiving node:
                    sent_messages = tf.concat(sent_messages, axis=0)
                    aggregated_received_messages = tf.unsorted_segment_sum(sent_messages,
                                                                           msg_targets_ta.read(substep_id),
                                                                           receiving_node_num_ta.read(substep_id))

                    # Collect old states for receiving nodes, and combine in RNN cell with incoming message
                    substep_receiving_nodes = receiving_nodes_ta.read(substep_id)
                    old_receiving_node_states = tf.gather(cur_node_states, substep_receiving_nodes)
                    aggregated_received_messages.set_shape([None, self.params['hidden_size']])
                    old_receiving_node_states.set_shape([None, self.params['hidden_size']])
                    substep_new_node_states = self.weights['rnn_cells'](aggregated_received_messages,
                                                                        old_receiving_node_states)[1]

                    # Write updated states back:
                    new_node_states_ta = new_node_states_ta.scatter(substep_receiving_nodes, substep_new_node_states)
                    return (substep_id + 1, new_node_states_ta)

                def is_done(substep_id, new_node_states_ta_unused):
                    return tf.logical_and(substep_id < self.params['propagation_substeps'],
                                          tf.greater(tf.shape(receiving_nodes_ta.read(substep_id))[0], 0))

                _, new_node_states_ta = tf.while_loop(cond=is_done,
                                                      body=do_substep,
                                                      loop_vars=[tf.constant(0), new_node_states_ta]
                                                     )

                cur_node_states = new_node_states_ta.stack(name="state_stack_round%i" % (prop_round,))

        return cur_node_states

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]

        # Sum up all nodes per graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 1]
        return tf.squeeze(graph_representations)  # [g]

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        processed_graphs = []
        for d in raw_data:
            prop_schedules = self.__graph_to_propagation_schedules(d['graph'])
            processed_graphs.append({"init": d["node_features"],
                                     "prop_schedules": prop_schedules,
                                     "target_values": [d["targets"][task_id][0] for task_id in self.params['task_ids']]})

        if is_training_data:
            np.random.shuffle(processed_graphs)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['target_values'][task_id] = None

        return processed_graphs

    def __tensorise_edge_sequence(self, edges)\
            -> Tuple[np.ndarray, List[List[np.ndarray]], List[List[np.ndarray]], List[np.ndarray]]:
        sending_nodes = []  # type: List[List[np.ndarray]]
        msg_targets = []  # type: List[List[np.ndarray]]
        receiving_nodes = []  # type: List[np.ndarray]
        all_nodes = set()
        for step_edges in edges:
            msg_targets_uniq = set(w for (_, __, w) in step_edges)
            recv_nodes = list(sorted(msg_targets_uniq))
            recv_nodes_to_uniq_id = {v: i for (i, v) in enumerate(recv_nodes)}

            sending_nodes_in_step = []
            msg_targets_in_step = []
            for target_e_typ in range(self.num_edge_types):
                sending_nodes_in_step.append(np.array([v for (v, e_typ, _) in step_edges if e_typ == target_e_typ], dtype=np.int32))
                msg_targets_in_step.append(np.array([recv_nodes_to_uniq_id[w] for (_, e_typ, w) in step_edges if e_typ == target_e_typ], dtype=np.int32))
            msg_targets.append(msg_targets_in_step)
            sending_nodes.append(sending_nodes_in_step)
            receiving_nodes.append(np.array(recv_nodes, dtype=np.int32))
            all_nodes.update(v for (v, _, __) in step_edges)
            all_nodes.update(w for (_, __, w) in step_edges)
        
        all_updated_nodes = set()
        all_updated_nodes.update(v for step_receiving_nodes in receiving_nodes
                                   for v in step_receiving_nodes)
        initial_nodes = list(sorted(all_nodes - all_updated_nodes))

        #initialised_nodes = set()
        #initialised_nodes.update(initial_nodes)
        #for step in range(len(receiving_nodes)):
        #    sent_nodes = set()
        #    for edge_typ in range(self.num_edge_types):
        #        sent_nodes.update(sending_nodes[step][edge_typ])
        #    for v in sent_nodes:
        #        assert v in initialised_nodes
        #
        #    for v in receiving_nodes[step]:
        #        assert v not in initialised_nodes
        #    initialised_nodes.update(receiving_nodes[step])

        return (np.array(initial_nodes, dtype=np.int32), sending_nodes, msg_targets, receiving_nodes)

    def __graph_to_propagation_schedules(self, graph)\
            -> List[Tuple[np.ndarray, List[List[np.ndarray]], List[List[np.ndarray]], List[np.ndarray]]]:
        num_incoming_edges = defaultdict(lambda: 0)
        outgoing_edges = defaultdict(lambda: [])
        # Compute number of incoming edges per node, and build adjacency lists:
        for (v, typ, w) in graph:
            num_incoming_edges[v] += 1
            num_incoming_edges[w] += 1
            edge_bwd_typ = typ if self.params['tie_fwd_bkwd'] else self.num_edge_types + typ
            outgoing_edges[v].append((v, typ, w))
            outgoing_edges[w].append((w, edge_bwd_typ, v))

        # Sort them, pick node with lowest number of incoming edges:
        tensorised_prop_schedules = []
        for prop_round in range(int(self.params['propagation_rounds'] / 2)):
            dag_seed = min(num_incoming_edges.items(), key=lambda t: t[1])[prop_round]
            node_depths = {}
            bfs_visit(outgoing_edges, node_depths, dag_seed, 0)

            # Now split edges into forward/backward sets, by using their depths.
            # Intuitively, a node with depth h will get updated in step h.
            max_depth = max(node_depths.values())
            assert(max_depth <= self.params['propagation_substeps'])
            fwd_pass_edges = [[] for _ in range(max_depth)]
            bwd_pass_edges = [[] for _ in range(max_depth)]
            for (v, typ, w) in graph:
                edge_bwd_type = typ if self.params['tie_fwd_bkwd'] else self.num_edge_types + typ
                v_depth = node_depths[v]
                w_depth = node_depths[w]
                if v_depth < w_depth:  # "Forward": We are going up in depth:
                    fwd_pass_edges[w_depth - 1].append((v, typ, w))
                    bwd_pass_edges[-v_depth - 1].append((w, edge_bwd_type, v))
                elif w_depth < v_depth:  # "Backward": We are going down in depth
                    fwd_pass_edges[v_depth - 1].append((w, edge_bwd_type, v))
                    bwd_pass_edges[-w_depth - 1].append((v, typ, w))
                else:
                    # We ignore self-loops:
                    assert v == w

            tensorised_prop_schedules.append(self.__tensorise_edge_sequence(fwd_pass_edges))
            tensorised_prop_schedules.append(self.__tensorise_edge_sequence(bwd_pass_edges))

        return tensorised_prop_schedules

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening graphs into a single one with multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_graph_nodes_list = []
            node_offset = 0

            # Collect all indices; we'll strip out the batch dimension with a np.concatenate along that axis at the end:
            batch_initial_nodes = [[] for _ in range(self.params['propagation_rounds'])
                                  ]  # type: List[List[np.ndarray]]              # (prop_round, batch, None)
            batch_sending_nodes = [[[[] for _ in range(self.num_edge_types)]
                                    for _ in range(self.params['propagation_substeps'])]
                                   for _ in range(self.params['propagation_rounds'])
                                  ]  # type: List[List[List[List[np.ndarray]]]]  # (prop_round, step, edge_typ, batch, None)
            batch_msg_targets = [[[[] for _ in range(self.num_edge_types)]
                                  for _ in range(self.params['propagation_substeps'])]
                                 for _ in range(self.params['propagation_rounds'])
                                ]  # type: List[List[List[List[np.ndarray]]]]    # (prop_round, step, edge_typ, batch, None)
            batch_receiving_nodes = [[[] for _ in range(self.params['propagation_substeps'])]
                                     for _ in range(self.params['propagation_rounds'])
                                    ]  # type: List[List[List[np.ndarray]]]      # (prop_round, step, batch, None)
            batch_receiving_node_num = [[0 for _ in range(self.params['propagation_substeps'])]
                                        for _ in range(self.params['propagation_rounds'])
                                       ]  # type: List[List[int]]                # (prop_round, step)

            msg_target_offsets = [[[0 for _ in range(self.num_edge_types)]
                                   for _ in range(self.params['propagation_substeps'])]
                                  for _ in range(self.params['propagation_rounds'])]

            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['num_nodes']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))

                # Combine the different propagation schedules:
                for prop_round in range(self.params['propagation_rounds']):
                    cur_prop_schedule = cur_graph['prop_schedules'][prop_round]
                    (graph_initial_nodes,
                     graph_sending_nodes,
                     graph_msg_targets,
                     graph_recv_nodes) = cur_prop_schedule
                    batch_initial_nodes[prop_round].append(graph_initial_nodes + node_offset)
                    for step in range(self.params['propagation_substeps']):
                        # Stop if we don't have that many steps:
                        if step >= len(graph_sending_nodes):
                            break

                        for e_typ in range(self.num_edge_types):
                            batch_sending_nodes[prop_round][step][e_typ].append(graph_sending_nodes[step][e_typ] + node_offset)
                            batch_msg_targets[prop_round][step][e_typ].append(graph_msg_targets[step][e_typ] + msg_target_offsets[prop_round][step][e_typ])
                            if len(graph_msg_targets[step][e_typ]) > 0:
                                msg_target_offsets[prop_round][step][e_typ] += max(graph_msg_targets[step][e_typ]) + 1  # ... 0-based indexing!
                        batch_receiving_nodes[prop_round][step].append(graph_recv_nodes[step] + node_offset)
                        batch_receiving_node_num[prop_round][step] += len(graph_recv_nodes[step])

                target_task_values = []
                target_task_mask = []
                for target_val in cur_graph['target_values']:
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
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list, axis=0),
                self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1,0]),
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
            }

            for prop_round in range(self.params['propagation_rounds']):
                batch_feed_dict[self.placeholders['initial_nodes'][prop_round]] = \
                    np.concatenate(batch_initial_nodes[prop_round], axis=0)
                for step in range(self.params['propagation_substeps']):
                    msg_targets = []
                    for edge_typ in range(self.num_edge_types):
                        raw_senders = batch_sending_nodes[prop_round][step][edge_typ]
                        batch_feed_dict[self.placeholders['sending_nodes'][prop_round][step][edge_typ]] = \
                            np.concatenate(raw_senders, axis=0) if len(raw_senders) > 0 else np.empty(shape=(0,),
                                                                                                      dtype=np.int32)
                        raw_targets = batch_msg_targets[prop_round][step][edge_typ]
                        msg_targets.extend(np.concatenate(raw_targets, axis=0) if len(raw_targets) > 0 else np.empty(shape=(0,),
                                                                                                                     dtype=np.int32))

                    batch_feed_dict[self.placeholders['msg_targets'][prop_round][step]] = \
                        np.array(msg_targets, dtype=np.int32)
                    raw_recvs = batch_receiving_nodes[prop_round][step]
                    batch_feed_dict[self.placeholders['receiving_nodes'][prop_round][step]] = \
                        np.concatenate(raw_recvs, axis=0) if len(raw_recvs) > 0 else np.empty(shape=(0,),
                                                                                              dtype=np.int32)
                batch_feed_dict[self.placeholders['receiving_node_num'][prop_round]] = \
                    np.array(batch_receiving_node_num[prop_round])

            #self.check_batch_invariants(batch_feed_dict)
            yield batch_feed_dict


    def check_batch_invariants(self, batch_feed_dict):
        for prop_round in range(self.params['propagation_rounds']):
            initialised_nodes = set()
            initialised_nodes.update(batch_feed_dict[self.placeholders['initial_nodes'][prop_round]])
            for step in range(self.params['propagation_substeps']):
                sending_nodes = set()
                for edge_typ in range(self.num_edge_types):
                    sending_nodes.update(batch_feed_dict[self.placeholders['sending_nodes'][prop_round][step][edge_typ]])
                for v in sending_nodes:
                    assert v in initialised_nodes

                recv_nodes = batch_feed_dict[self.placeholders['receiving_nodes'][prop_round][step]]
                for v in recv_nodes:
                    assert v not in initialised_nodes
                initialised_nodes.update(recv_nodes)

def main():
    args = docopt(__doc__)
    try:
        model = AsyncGGNNChemModel(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
