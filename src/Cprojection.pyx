import cv2
import numpy as np
from numpy import linalg as LA
import math

cimport numpy as np


def project_nneg(x, L1, L2, verbose=True):
	"""
	Given any vector x, find the closest (in the euclidean sense) 
	non-negative vector s with a given L1 norm and a given L2 norm.
	(note: the specified L1 and L2 norm is not completely arbitrary)
	input args: x -- input vector 
				L1 -- specified L1 norm
				L2 -- specified L2 norm
				verbose -- set true for detailed log
	output args: s: projected non-negative vector
	"""

	cdef int dim = len(x)
	assert(L1 >= L2 and L1 <= math.sqrt(dim)*L2)
	if verbose == True:
		print "Begin projection: \n Dimention: %d;" % dim
		print "L1: %f, L2: %f;\n target_L1: %f, target_L2:%f" \
				% (LA.norm(x, 1), LA.norm(x, 2), L1, L2)

	x = x.reshape((dim, 1))
	cdef np.ndarray[np.int64_t, ndim=1] zero_idx
	zero_idx=np.array([], dtype=np.int64)
	cdef int iter = 0
	cdef np.ndarray[np.float_t, ndim=2] mid, s, w
	cdef float a, b, c, delta, alpha, tmpsum
	s = x + (L1 - np.sum(x)) / dim;

	while(1):
		mid = np.zeros((dim, 1), dtype=np.float)
		mid += L1 / (dim - len(zero_idx));
		mid[zero_idx] = 0
		w = s - mid
		a_raw = np.sum(w**2);
		b_raw = 2 * np.dot(w.reshape(dim,), s.reshape(dim,));
		c_raw = np.sum(s**2) - pow(L2, 2);
		a = a_raw/a_raw
		b = b_raw/a_raw
		c = c_raw/a_raw
		delta = pow(b, 2) - 4 * a * c;
		assert(delta >= 0)
		alpha = (-b + math.sqrt(delta)) / (2 * a);
		s = s + alpha * w;
		iter += 1

		if np.array(np.where(s<0)).size == 0:
			#assert(abs(LA.norm(s, 1) - L1) <= 0.5 and abs(LA.norm(s, 2) - L2) <= 0.5)
			if verbose == True:
				print "Projection finished:\n final_L1: %f, final_L2: %f" \
						% (LA.norm(s, 1), LA.norm(s, 2))
				print "%d iterations used for the projection\n" % iter 
			return s

		# if after 100 iterations we still do not get the desired projection
		# we set negative elements to 0 and return s
		if iter >= 500:
			np.where(s<0, abs(s), s)
			if verbose == True:	
				print "WARNING: after 500 iterations still not project to nneg, force nneg then"
				print "Projection finished:\n final_L1: %f, final_L2: %f" \
						% (LA.norm(s, 1), LA.norm(s, 2))
				print "%d iterations used for the projection\n" % iter
			return s

		zero_idx = np.where(s < 0)[0]
		s[zero_idx] = 0

		tmpsum = sum(s)
		s = s + (L1 - tmpsum) / (dim - len(zero_idx));


