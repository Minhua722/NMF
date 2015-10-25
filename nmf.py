from numpy import linalg as LA
import numpy as np
import math

def L1_for_sparseness(D, L2, sparseness):
	"""
	For a vector with dim D, find qualified L1 norm 
	given L2 norm and desired sparseness
	sparseness = (sqrt(n) - L1/L2) / (sqrt(n)-1)
	input args: D -- dimension
				L2 -- specified L2 norm
				sparseness - specified sparseness
	output args: L1 -- qualified L1 norm
	"""

	L1 = (math.sqrt(D) - sparseness * (math.sqrt(D) - 1)) * L2;
	return L1

def project_nneg(x, L1, L2):
	"""
	Given any vector x, find the closest (in the euclidean sense) 
	non-negative vector s with a given L1 norm and a given L2 norm.
	input args: x -- input vector 
				L1 -- specified L1 norm
				L2 -- specified L2 norm
	output args: s: projected non-negative vector
	"""
	
	dim = len(x)
	print "Begin projection: \n Dimention: %d;\n L1: %f, L2: %f;\n target_L1: %f, target_L2:%f" % (dim, LA.norm(x, 1), LA.norm(x, 2), L1, L2)
	x = x.reshape((dim, 1))
	zero_idx = list()
	iter = 0
	s = x + (L1 - np.sum(x)) / dim;

	while(1):

		mid = np.zeros((dim, 1), dtype=np.float)
		mid += L1 / (dim - len(zero_idx));
		set_zeros(mid, zero_idx);
		w = s - mid
		a = np.sum(w**2);
		b = 2 * np.dot(w.reshape(dim,), s.reshape(dim,));
		c = np.sum(s**2) - pow(L2, 2);
		delta = pow(b, 2) - 4 * a * c;
   		assert(delta >= 0)
		alpha = (-b + math.sqrt(delta)) / (2 * a);
		s = s + alpha * w;
		iter += 1
		if check_all_nneg(s):
			print "%d iterations used for the projection\n" % iter 
			return s
		
		zero_idx = np.where(s < 0)
		set_zeros(s, zero_idx)
		
		tmpsum = sum(s)
		s = s + (L1 - tmpsum) / (dim - len(zero_idx));


def set_zeros(vec, idx):
	"""
	Set a particular group of indice in an array with element 0
	input args: vec -- input vector
				idx -- indices for which we want to set its element 0
	"""

	for i in idx:
		vec[i] = 0


def check_all_nneg(vec):
	"""
	Check if every element in the input vector is non-negative
	input args: vec -- input vector
	output: true/false
	"""
		
	for ele in vec:
		if ele < 0:
			return False
	return True
