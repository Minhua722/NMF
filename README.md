# NMF for face recognition
In this project, we implement NMF algorithms with sparseness constraint, label constraint as well as the basic NMF. 
A face recognition system is built using features extracted from different NMF algorithms on the AR face database


# Before running experiment
The module Cprojection need to be compiled using Cython
cd src/
python setup.py build_ext --inplace

This compile process take the code Cprojection.pyx and genenrate Cprojection.c, Cprojection.so in the current directory


# How to use
PCA and NMF are tested on 3 datasets: 
cbcl, orl, ar, 
which are implemented in egs/cbcl, egs/orl, egs/ar respectively

* cbcl
cd egs/cbcl
bash run.sh

* orl
cd egs/orl
bash run.sh

* ar
cd egs/ar
bash run.sh [num_man] [num_woman] [num_basis]

note: need to specify directory of data at the beginning of each run.sh experiment scripts
