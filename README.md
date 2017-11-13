Gated Graph Neural Networks
====
This repository contains two implementations of the Gated Graph Neural Networks
of [Li _et al_](https://arxiv.org/abs/1511.05493) for learning properties of chemical molecules. The inspiration for this application comes from [Gilmer _et al_](https://arxiv.org/abs/1704.01212)

This code was tested in Python 3.5 with TensorFlow 1.3.

To run the code `docopt` is also necessary.

Data Extraction
---
To download the related data run `get_data.py`. It requires the python package `rdkit` within the Python package
environment. For example, this can be obtained by
```
conda install -c rdkit rdkit
```

Running GGNN Training
---
We provide two versions of GGNNs. The dense version is faster for small or dense graphs,
including the molecules dataset. In contrast, the sparse version is faster for large
and sparse graphs, especially in cases where representing a dense representation of the
adjacency matrix would result in prohibitively large memory usage.


To run the dense GGNN,
```
python3 ./chem_tensorflow_dense.py
```


To run the sparse version of GGNN use,
```
python3 ./chem_tensorflow_sparse.py
```
# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
