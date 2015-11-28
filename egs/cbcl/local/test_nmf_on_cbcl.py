#!/usr/bin/env python

import cv2
import numpy as np
from sklearn.decomposition import PCA
import argparse
import math
import pickle

import sys, os
from nmf_support import *

if __name__ == '__main__': 

	#------------------------------------------------------
	# Args parser
	#------------------------------------------------------

	parser = argparse.ArgumentParser(description='test NMF on CBCL database')
	parser.add_argument('--num_basis', '-n', 
			action='store', type=int, default=36, 
			help='number of basis, default=36')
	parser.add_argument('--num_iterations', '-i',
			action='store', type=int, default=500,
			help='number of iterations. (default: 500)')
	parser.add_argument('--input_path', '-in',
			action='store', type=str, required=True,
			help='path to dataset directory')
	parser.add_argument('--output_path', '-out',
			action='store', type=str, required=True,
			help='directory to store result')

	args = parser.parse_args()
	data_dir = args.input_path
	exp_dir = args.output_path
	num_iter = args.num_iterations
	num_basis = args.num_basis
	
	print "Apply NMF (no constraint)"
	nmf_dir = "%s/nmf_standard" % exp_dir
	if not os.path.exists(nmf_dir):
		os.makedirs(nmf_dir)
	os.system("python local/factorizer.py -in %s -out %s -i %d -n %d" \
			% (data_dir, nmf_dir, num_iter, num_basis))
	print "Basis computed from NMF stored in %s\n" % nmf_dir

	print "Apply NMF (basis constrained to sparseness 0.7)"
	nmf_dir = "%s/nmf_W0.7" % exp_dir
	if not os.path.exists(nmf_dir):
		os.makedirs(nmf_dir)
	os.system("python local/factorizer.py -in %s -out %s -i %d -n %d -W 0.7" \
			% (data_dir, nmf_dir, num_iter, num_basis))
	print "Basis stored in %s\n" % nmf_dir

	print "Apply NMF (basis constrained to sparseness 0.3)"
	nmf_dir = "%s/nmf_W0.3" % exp_dir
	if not os.path.exists(nmf_dir):
		os.makedirs(nmf_dir)
	os.system("python local/factorizer.py -in %s -out %s -i %d -n %d -W 0.3" \
			% (data_dir, nmf_dir, num_iter, num_basis))
	print "Basis stored in %s\n" % nmf_dir

	print "Apply NMF (coef constrained to sparseness 0.7)"
	nmf_dir = "%s/nmf_H0.7" % exp_dir
	if not os.path.exists(nmf_dir):
		os.makedirs(nmf_dir)
	os.system("python local/factorizer.py -in %s -out %s -i %d -n %d -H 0.7" \
			% (data_dir, nmf_dir, num_iter, num_basis))
	print "Basis stored in %s\n" % nmf_dir

