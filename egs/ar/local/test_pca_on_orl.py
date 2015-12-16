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

	parser = argparse.ArgumentParser(description='test PCA and NMF on CBCL database')
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
	
	print "Apply PCA"
	pca_dir = "%s/pca" % exp_dir
	if not os.path.exists(pca_dir):
		os.makedirs(pca_dir)
	os.system("python local/eigenfaces.py -in %s -out %s -n %d" \
			% (data_dir, pca_dir, num_basis))
	print "Basis computed from PCA stored in %s\n" % pca_dir

