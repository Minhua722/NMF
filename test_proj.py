#!/usr/bin/python


import numpy as np
from numpy import linalg as LA
import math

from nmf_support import *

if __name__ == "__main__":

	# test projection of vectors with different dimensions
	Dims = [10, 50, 100, 1000]
	for dim in Dims:
		x = np.random.normal(0, 1, dim)
		target_L2 = LA.norm(x, 2)
		#target_L2 = 1
		target_L1 = L1_for_sparseness(dim, target_L2, 0.8)
		assert(target_L1 >= target_L2 and target_L1 <= math.sqrt(dim)*target_L2)
	
		s = project_nneg(x, target_L1, target_L2, True)
