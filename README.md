# Gated Graph Neural Networks
This repository contains two implementations of the Gated Graph Neural Networks
of [Li et al. 2015](https://arxiv.org/abs/1511.05493) for learning properties of chemical molecules.
The inspiration for this application comes from [Gilmer et al. 2017](https://arxiv.org/abs/1704.01212).

This code was tested in Python 3.5 with TensorFlow 1.3. To run the code `docopt` is also necessary.

This code is maintained by the [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/) project at Microsoft Research, Cambridge, UK.

## Data Extraction
To download the related data run `get_data.py`. It requires the python package `rdkit` within the Python package
environment. For example, this can be obtained by
```
conda install -c rdkit rdkit
```

## Running Graph Neural Network Training
We provide four versions of Graph Neural Networks: Gated Graph Neural Networks (one implementation using dense
adjacency matrices and a sparse variant), Asynchronous Gated Graph Neural Networks, and Graph Convolutional
Networks (sparse).
The dense version is faster for small or dense graphs, including the molecules dataset (though the difference is
small for it). In contrast, the sparse version is faster for large and sparse graphs, especially in cases where
representing a dense representation of the adjacency matrix would result in prohibitively large memory usage.
Asynchronous GNNs do not propagate information from all nodes to all neighbouring nodes at each timestep;
instead, they follow an update schedule such that messages are propagated in sequence. Their implementation
is far more inefficient (due to the small number of updates at each step), but a single propagation round
(i.e., performing each propagation step along a few edges once) can suffice to propagate messages across a
large graph.

To run dense Gated Graph Neural Networks, use
```
python3 ./chem_tensorflow_dense.py
```

To run sparse Gated Graph Neural Networks, use
```
python3 ./chem_tensorflow_sparse.py
```

To run sparse Graph Convolutional Networks (as in [Kipf et al. 2016](https://arxiv.org/abs/1609.02907)), use
```
python3 ./chem_tensorflow_gcn.py
```

Finally, it turns out that the extension of GCN to different edge types is a variant of GGNN, and you can run
GCN (as in [Schlichtkrull et al. 2017](https://arxiv.org/abs/1703.06103)) by calling
```
python3 ./chem_tensorflow_sparse.py --config '{"use_edge_bias": false, "use_edge_msg_avg_aggregation": true, "tie_gnn_layers": false, "graph_rnn_cell": "RNN", "graph_rnn_activation": "ReLU"}'
```

To run asynchronous Gated Graph Neural Networks, use
```
python3 ./chem_tensorflow_async.py
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
