#!/usr/bin/env/python

from typing import List, Tuple, Dict, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np

from utils import MLP, ThreadedIterator

from utils import MLP, ThreadedIterator, SMALL_NUMBER

class ChemModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.0001,
            'clamp_gradient_norm': 1.0,
            'dropout_keep_prob': 1.0,

            'hidden_size': 100,
            'num_timesteps': 4,

            'tie_fwd_bkwd': True,
            'task_ids': [0],
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data("molecules_train.json")
        self.valid_data = self.load_data("molecules_valid.json")

        # Build the actual model
        self.placeholders = {}
        self.weights = {}
        self.ops = {}
        self.make_model()

    def load_data(self, file_name):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any]) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        self.placeholders['dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
        self.prepare_specific_model()

        # This does the actual graph work:
        self.ops['final_node_representations'] = self.compute_final_node_representations()

        self.ops['loss'] = 0
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                       self.placeholders['dropout_keep_prob'])
                self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                            self.placeholders['dropout_keep_prob'])
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id])
                diff = computed_values - self.placeholders['target_values'][internal_id,:]
                self.ops['accuracy_task%i' % task_id] = tf.reduce_mean(tf.abs(diff))
                self.ops['loss'] += tf.reduce_mean(0.5 * diff ** 2)

        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'])
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, sess: tf.Session, epoch_name: str, data, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=3)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['dropout_keep_prob']] = self.params['dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]
            result = sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               num_graphs,
                                                                               loss / processed_graphs),
                  end='\r')

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracies, error_ratios, instance_per_sec

    def train(self):
        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        log_to_save = []
        total_time_start = time.time()
        (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
        for epoch in range(1, self.params['num_epochs']):
            print("== Epoch %i" % epoch)
            train_loss, train_accs, train_errs, train_speed = self.run_epoch(sess, "epoch %i (training)" % epoch,
                                                                             self.train_data, True)
            accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
            errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
            print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                    accs_str,
                                                                                                    errs_str,
                                                                                                    train_speed))
            valid_loss, valid_accs, valid_errs, valid_speed = self.run_epoch(sess, "epoch %i (validation)" % epoch,
                                                                             self.valid_data, False)
            accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
            errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
            print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                    accs_str,
                                                                                                    errs_str,
                                                                                                    valid_speed))

            epoch_time = time.time() - total_time_start
            log_entry = {
                'epoch': epoch,
                'time': epoch_time,
                'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
            }
            log_to_save.append(log_entry)
            with open(self.log_file, 'w') as f:
                json.dump(log_to_save, f, indent=4)

            val_acc = np.sum(valid_accs)  # type: float
            if val_acc < best_val_acc:
                print("  (Best epoch so far, val. acc decreased to %.5f from %.5f)" % (val_acc, best_val_acc))
                best_val_acc = val_acc
                best_val_acc_epoch = epoch
            elif epoch - best_val_acc_epoch >= self.params['patience']:
                print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                break