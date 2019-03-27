#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_dense.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
"""

from typing import Sequence, Any
from docopt import docopt
from collections import defaultdict
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json

from chem_tensorflow import ChemModel
from utils import glorot_init


def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    for src, e, dest in graph:
        amat[e-1, dest, src] = 1
        amat[e-1 + bwd_edge_offset, src, dest] = 1
    return amat


'''
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edge types (4)
v:      number of vertices per graph in this batch
h:      GNN hidden size
'''

class DenseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
                        'batch_size': 256,
                        'graph_state_dropout_keep_prob': 1.,
                        'task_sample_ratios': {},
                        'use_edge_bias': True,
                        'edge_weight_dropout_keep_prob': 1
                      })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # inputs
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])     # [b, e, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])         # [e, b, v, v]


        # weights
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        h = self.placeholders['initial_node_representation']                                                # [b, v, h]
        h = tf.reshape(h, [-1, h_dim])

        with tf.variable_scope("gru_scope") as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type],
                                                   keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])) # [b*v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                       # [b, v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'][edge_type]                                         # [b, v, h]
                    if edge_type == 0:
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])                                                        # [b*v, h]

                h = self.weights['node_gru'](acts, h)[1]                                                    # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [b x v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis = 2)        # [b, v, 2h]
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])                           # [b*v, 2h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])                                       # [b*v, h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)           # [b*v, 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices']])                  # [b, v]
        masked_gated_outputs = gated_outputs * self.placeholders['node_mask']                               # [b x v]
        output = tf.reduce_sum(masked_gated_outputs, axis = 1)                                              # [b]
        self.output = output
        return output

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        if bucket_sizes is None:
            bucket_sizes = np.array(list(range(4, 28, 2)) + [29])
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features"][0])
        for d in raw_data:
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            n_active_nodes = len(d["node_features"])
            bucketed[chosen_bucket_idx].append({
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types, self.params['tie_fwd_bkwd']),
                'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                'mask': [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def pad_annotations(self, annotations):
        return  np.pad(annotations,
                       pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                       mode='constant')


    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'node_mask': [], 'task_masks': []}
        for d in elements:
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']:
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)

        return batch_data


    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['init'])
            initial_representations = batch_data['init']
            initial_representations = self.pad_annotations(initial_representations)

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: initial_representations,
                self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1,0]),
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                self.placeholders['node_mask']: batch_data['node_mask'],
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob
            }

            bucket_counters[bucket] += 1

            yield batch_feed_dict

    def evaluate_one_batch(self, initial_node_representations, adjacency_matrices, node_masks=None):
        num_vertices = len(initial_node_representations[0])
        if node_masks is None:
            node_masks = []
            for r in initial_node_representations:
                node_masks.append([1. for _ in r] + [0. for _ in range(num_vertices - len(r))])
        batch_feed_dict = {
            self.placeholders['initial_node_representation']: self.pad_annotations(initial_node_representations),
            self.placeholders['num_graphs']: len(initial_node_representations),
            self.placeholders['num_vertices']: len(initial_node_representations[0]),
            self.placeholders['adjacency_matrix']: adjacency_matrices,
            self.placeholders['node_mask']: node_masks,
            self.placeholders['graph_state_keep_prob']: 1.0,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
            self.placeholders['edge_weight_dropout_keep_prob']: 1.0
        }

        fetch_list = self.output
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result

    def example_evaluation(self):
        ''' Demonstration of what test-time code would look like
        we query the model with the first n_example_molecules from the validation file
        '''
        n_example_molecules = 10
        with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]

        for mol in example_molecules:
            print(mol['targets'])

        example_molecules, _, _ = self.process_raw_graphs(example_molecules, 
            is_training_data=False, bucket_sizes=np.array([29]))
        batch_data = self.make_batch(example_molecules[0])
        print(self.evaluate_one_batch(batch_data['init'], batch_data['adj_mat']))

        



def main():
    args = docopt(__doc__)
    try:
        model = DenseGGNNChemModel(args)

        if args['--evaluate']:
            model.example_evaluation()
        else:
            model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
