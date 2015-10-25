#!/usr/bin/python
import numpy as np
from numpy import linalg as LA
import math

from nmf import *

if __name__ == "__main__":

	# test projection of vectors with different dimensions
	Dims = [10, 50, 100, 1000]
	for dim in Dims:
		x = np.random.normal(0, 1, dim)
		target_L2 = LA.norm(x, 2)
		target_L1 = L1_for_sparseness(dim, target_L2, 0.2)
		assert(target_L1 >= target_L2 and target_L1 <= math.sqrt(dim)*target_L2)
	
		s = project_nneg(x, target_L1, target_L2)
		new_L1 = LA.norm(s, 1)
		new_L2 = LA.norm(s, 2)
		print "Projection finished:\n final_L1: %f, final_L2: %f\n" % (new_L1, new_L2)
	
		assert (abs(new_L1 - target_L1) <= 0.5 and abs(new_L2 - target_L2) <= 0.5)
