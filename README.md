# NMF
non-negative matrix factorization

# Before running experiment
The module Cprojection are compiled using Cython
python setup.py build_ext --inplace
This compile process take the code Cprojection.pyx and genenrate Cprojection.c, Cprojection.so in the current directory


# How to use 
./factorizer.py -in ../../cbcl_faces/train/face -out output

To see detailed options available please use:
./factorizer.py -h 
