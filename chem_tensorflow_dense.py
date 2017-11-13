#!/usr/bin/env/python
'''
Usage:
    chem_tensorflow_dense.py [options]

Options:
    -h --help                Show this screen.
    --config CONFIG          config file path
    --out NAME               out file name
    --out_dir NAME           out dir name
    --data_dir NAME          data dir name
'''
from __future__ import print_function
from docopt import docopt
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import os
import json

import sys, traceback
np.random.seed(0)
tf.set_random_seed(0)

def load_data(file_name, data_dir, tie_fwd_bkwd=False, shuffle=True):
    full_path = os.path.join(data_dir, file_name)
    print("loading data from: ", full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)
    max_n_vertices, n_fwd_edges = get_graph_sizes(data)
    print("n_vertex: ", max_n_vertices, "n_edges: ", n_fwd_edges)
    x_dim = len(data[0]["node_features"][0])

    buckets = np.array(list(range(4,28,2)) + [29])
    bucketed = defaultdict(list)
    for d in data:
        b = np.argmax(buckets > max([v for e in d['graph'] for v in [e[0], e[2]]]))
        bucketed[b].append({
            'adj_mat': graph_to_adj_mat(d['graph'], buckets[b], n_fwd_edges, tie_fwd_bkwd),
            'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in range(buckets[b] - len(d["node_features"]))],
            'label': d["targets"][0][0]
        })

    if shuffle:
        np.random.shuffle(data)
    return bucketed, buckets, n_fwd_edges, x_dim

def get_graph_sizes(dataset):
    max_n_vertices = 0
    n_fwd_edges = 0
    for g in dataset:
        max_n_vertices = max(max_n_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
        n_fwd_edges = max(n_fwd_edges, max([e[1] for e in g['graph']]))
    return max_n_vertices + 1, n_fwd_edges

def graph_to_adj_mat(graph, max_n_vertices, n_fwd_edges, tie_fwd_bkwd=True):
    amat_width = max_n_vertices if tie_fwd_bkwd else 2*max_n_vertices
    n_edge_types = 4 if tie_fwd_bkwd else 2*4
    amat = np.zeros((n_edge_types, max_n_vertices, amat_width))
    for src, e, dest in graph:
        amat[e-1, dest, src] = 1
        offset = 0 if tie_fwd_bkwd else 4
        amat[e + offset-1, src, dest] = 1
    return amat

def get_batch(data, start, stop):
    elements = data[start:stop]
    batch = {'adj_mat': [], 'init': [], 'label': []}
    for d in elements:
        batch['adj_mat'].append(d['adj_mat'])
        batch['init'].append(d['init'])
        batch['label'].append(d['label'])
    return batch

def init_weights(shape):
    return np.sqrt(6.0/(shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) -1)

class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1],dims[1:]))
        bias_sizes = dims[1:]
        weights = [tf.Variable(init_weights(s)) for s in weight_sizes]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32)) for s in weight_sizes]
        network_params = {"weights" : weights, "biases" : biases}
        return network_params
    
    def __call__(self, inputs):
        acts = inputs
        for W,b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, W) + b
            acts = tf.nn.dropout(tf.nn.relu(hid), self.dropout_keep_prob)
        last_hidden = hid
        return last_hidden


class GGNN():
    def __init__(self, params):
        self.params = params
        self.n_vertex = tf.placeholder(tf.int32, None)
        tie_fwd_bkwd = params['tie_fwd_bkwd']
        b = params['batch_size']
        v = self.n_vertex
        self.n_edge_types = params['n_edge_types'] * (1 if tie_fwd_bkwd else 2)
        x_dim = params['annotation_size']
        h_dim = params['hidden_size']

        # inputs
        self.x = tf.placeholder(tf.float32, None)               # [b, v, x_dim])
        self.adj_mat_input = tf.placeholder(tf.float32, None)   # [b x e x v x v]
        self.a = tf.transpose(self.adj_mat_input, [1,0,2,3])    # [e x b x v x v]
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, None)

        self.regression_gate = MLP(2*h_dim, 1, [], self.dropout_keep_prob)
        self.regression_transform = MLP(h_dim, 1, [], self.dropout_keep_prob)

        # weights
        self.edge_weights = tf.Variable(init_weights([self.n_edge_types, h_dim, h_dim]))
        self.edge_biases = tf.Variable(np.zeros([self.n_edge_types, 1, h_dim]).astype(np.float32))
        with tf.variable_scope("gru_scope") as scope:
            self.node_gru = tf.contrib.rnn.GRUCell(h_dim)
        self.weights = {
            "edge_weights": self.edge_weights,
            "edge_biases": self.edge_biases,
            "gru": self.node_gru
        }       

        h = tf.pad(self.x, [[0,0],[0,0],[0,h_dim-x_dim]]) 

        last_h = self.unroll(h, params['sequence_length'])
        self.output = self.gated_regression(last_h)
        self.make_training_parts()

    def unroll(self, init_h, n_unroll):
        b = self.params['batch_size']
        v = self.n_vertex
        h_dim = self.params['hidden_size']

        h = init_h                       # [b x v x h]
        h = tf.reshape(h, [-1, h_dim])   # [b*v x h]
        self.init_h = init_h 

        self.a = tf.reshape(self.a, [self.n_edge_types, b, v, v])

        bias = []
        for a in tf.unstack(self.a, axis=0):
            summed_a = tf.reshape(tf.reduce_sum(a, axis=-1), [-1, 1])    # [b*v x 1]
            bias.append(tf.matmul(summed_a, self.edge_biases[0]))        # [b*v x h]
        with tf.variable_scope("gru_scope") as scope:
            for i in range(n_unroll):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for i in range(self.n_edge_types):
                    m = tf.matmul(h, self.edge_weights[i]) + bias[i]     # [b*v x h]
                    m = tf.reshape(m, [-1, v ,h_dim])                    # [b x v x h]
                    if i == 0:
                        acts = tf.matmul(self.a[i], m) 
                    else:
                        acts += tf.matmul(self.a[i], m)
                acts = tf.reshape(acts, [-1, h_dim])                     # [b*v x h]
                h = self.node_gru(acts, h)[0]                            # [b*v x h]
            last_h = tf.reshape(h, [-1, self.n_vertex, self.params['hidden_size']])
        return last_h

    def gated_regression(self, last_h):
        # last_h: [b x v x h]
        if int(tf.__version__[0]) < 1:
            gate_input = tf.concat(concat_dim=2, values=[last_h, self.init_h])                              # [b x v x 2h]
        else:
            gate_input = tf.concat([last_h, self.init_h], axis = 2)                                         # [b x v x 2h]
        gate_input = tf.reshape(gate_input, [-1, 2*self.params["hidden_size"]])                             # [(b*v) x 2h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])                                       # [(b*v) x h]
        gated_outputs = tf.nn.sigmoid(self.regression_gate(gate_input)) * self.regression_transform(last_h) # [(b*v) x 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, self.n_vertex])                                      # [b x v]
        output = tf.reduce_sum(gated_outputs, axis = 1)                                                     # [b]
        return output
    
    def make_training_parts(self):
        self.target_node = tf.placeholder(tf.float32, [self.params['batch_size']])
        diff = self.output - self.target_node
        self.loss_node = tf.reduce_mean(0.5 * (diff)**2)
        self.accuracy_node = tf.reduce_mean(tf.abs(diff))
        
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss_node)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))

        self.train_node = optimizer.apply_gradients(clipped_grads)
      
def default_params():
    return {
        'batch_size': 256,          'n_epochs': 300,            
        'learning_rate': 0.0001,    'final_learning_rate': 0.0001,
        'clamp_gradient_norm': 1.0, 'hidden_size': 100,
        'sequence_length': 4,       'out': 'log.json',
        'device': 0,                'tie_fwd_bkwd': True,
        'dropout_keep_prob': 1.0,    
    }

def make_params(args):
    params = default_params()
    if '--config' in args and args['--config'] is not None:
        with open(args['--config'], 'r') as f:
            config_dict = json.load(f)
        for opt, value in config_dict.items():
            params[opt] = value
    print(params)
    return params

def get_data(args, params):
    data_dir = ''
    if '--data_dir' in args and args['--data_dir'] is not None:
        data_dir = args['--data_dir']
    data = {}
    data['train'], buckets, n_fwd_edges_train, x_dim = load_data(
        "molecules_train.json", data_dir, params['tie_fwd_bkwd'], shuffle=False)
    data['valid'], buckets, n_fwd_edges_test, x_dim = load_data(
        "molecules_valid.json", data_dir, params['tie_fwd_bkwd'], shuffle=False)
    n_fwd_edges = max([n_fwd_edges_train, n_fwd_edges_test])  
    if not params['tie_fwd_bkwd']:
        n_fwd_edges = 2 * n_fwd_edges  
    params.update({
        'n_edge_types': n_fwd_edges,
        'annotation_size': x_dim,
        'data_dir': data_dir
        })
    return data, buckets

def update_output_path(args, params):
    out_dir = ''
    if '--out_dir' in args and args['--out_dir'] is not None:
        out_dir = args['--out_dir']
    if '--out' in args and args['--out'] is not None:
        out = args['--out']
        params['out'] = os.path.join(out_dir, out)

def training_loop(sess, model, data, params, bucket_at_step, bucket_sizes, is_training):
    chemical_accuracy = [0.066513725]
    
    loss = 0
    accuracy = 0
    instances = 0
    start_time = time.time()
    bucket_counters = defaultdict(int)
    for step in range(len(bucket_at_step)):
        bucket = bucket_at_step[step]
        batch = get_batch(data[bucket], bucket_counters[bucket]*params['batch_size'], (bucket_counters[bucket]+1)*params['batch_size'])
        instances += len(batch['init'])
        feed_dict = {
            model.x: batch['init'],
            model.adj_mat_input: batch['adj_mat'], 
            model.target_node: batch['label'],
            model.n_vertex: bucket_sizes[bucket]
        }
        if is_training:
            feed_dict[model.dropout_keep_prob] = params['dropout_keep_prob']
            fetch_list = [model.loss_node, model.accuracy_node, model.train_node]
        else:
            fetch_list = [model.loss_node, model.accuracy_node]
        result = sess.run(fetch_list, feed_dict)
        loss += result[0] / len(bucket_at_step)
        accuracy += result[1] / len(bucket_at_step)

        if step % 100 == 0:
            print(instances, result[0])
        
        bucket_counters[bucket] += 1

    error_ratio = accuracy / chemical_accuracy[0]

    instance_per_sec = instances / (time.time() - start_time)
    print("loss: %s | error_ratio: %s | instances/sec: %s" % (loss, error_ratio, instance_per_sec))
    return instances / (time.time() - start_time), loss, accuracy



def main():
    args = docopt(__doc__)
    params = make_params(args)
    data, buckets = get_data(args, params)
    update_output_path(args, params)

    device_string = '/cpu:0' if params["device"] < 0 else '/gpu:%s' % str(params["device"])
    with tf.device(device_string):
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            model = GGNN(params)

        config = tf.ConfigProto(allow_soft_placement=True)
        if params["device"] >= 0:
            config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        init_op = tf.group(tf.global_variables_initializer(), 
                        tf.local_variables_initializer())
        sess.run(init_op)

        bucket_at_step = {}
        for section in ['train', 'valid']:
            bucket_at_step[section] = [[bucket for _ in range(len(d) // params['batch_size'])] for bucket, d in data[section].items()]
            bucket_at_step[section] = [x for y in bucket_at_step[section] for x in y]


        log_to_save = []
        total_time_start = time.time()
        total_time = 0
        val_acc = 0
        for epoch in range(1, params['n_epochs']):
            log_entry = {}

            np.random.shuffle(bucket_at_step['train'])
            np.random.shuffle(bucket_at_step['valid'])
            for _, d in data['train'].items():
                np.random.shuffle(d)

            print('epoch', epoch, 'train ',)
            train_instances_per_s, train_loss, train_acc = training_loop(
                sess, model, data['train'], params, bucket_at_step['train'], buckets, True)
            if True:
                print('epoch', epoch, 'valid ',)
                val_instances_per_s, val_loss, val_acc = training_loop(
                sess, model, data['valid'], params, bucket_at_step['valid'], buckets, False)
            else:
                val_instances_per_s = val_loss = -1

            total_time = time.time() - total_time_start
            log_entry['epoch'] = epoch;            log_entry['time'] = total_time;            log_entry['train_instances_per_s'] = train_instances_per_s
            log_entry['train_loss'] = train_loss;  log_entry['train_acc'] = train_acc;        log_entry['val_instances_per_s'] = val_instances_per_s
            log_entry['val_loss'] = val_loss;      log_entry['val_acc'] = val_acc; 
            log_to_save.append(log_entry)
            with open(params['out'], 'w') as f:
                json.dump(log_to_save, f, indent=4)


if __name__ == "__main__":
    main()
