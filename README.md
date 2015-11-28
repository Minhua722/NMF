# NMF
non-negative matrix factorization

# Before running experiment
The module Cprojection need to be compiled using Cython
cd src/
python setup.py build_ext --inplace

This compile process take the code Cprojection.pyx and genenrate Cprojection.c, Cprojection.so in the current directory


# How to use
PCA and NMF are tested on 3 datasets: 
cbcl, orl, ar, 
which are implemented in egs/cbcl, egs/orl, egs/ar respectively
