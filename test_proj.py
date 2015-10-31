#!/usr/bin/python

import numpy as np
from numpy import linalg as LA
import math

import nmf_support
import pstats, cProfile

import nmf_support
import Cprojection

if __name__ == "__main__":

	# profile pojection of a single vector 
#	dim = 10000
#	x = np.random.normal(0, 1, dim)
#	target_L2 = LA.norm(x, 2)
#	target_L1 = nmf_support.L1_for_sparseness(dim, target_L2, 0.8)
#	assert(target_L1 >= target_L2 and target_L1 <= math.sqrt(dim)*target_L2)    
#	cProfile.runctx("Cprojection.project_nneg(x, target_L1, target_L2, True)", globals(), locals(), "Profile.prof")
#	s = pstats.Stats("Profile.prof")
#	s.strip_dirs().sort_stats("time").print_stats()


	# profile projection of rows of a whole matrix
	W, H = nmf_support.initial_WH(400, 30, 30, 1000)
	cProfile.runctx("nmf_support.project_matrix_row(H, 0.8, 'unit')", globals(), locals(), "Profile.prof")
	s = pstats.Stats("Profile.prof")
	s.strip_dirs().sort_stats("time").print_stats()
