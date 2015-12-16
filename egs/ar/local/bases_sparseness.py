#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import math
import pickle

from sklearn.decomposition import PCA
from nmf_support import *

import sys, os 


if __name__ == '__main__': 

	#------------------------------------------------------
	# Args parser
	#------------------------------------------------------

	parser = argparse.ArgumentParser(description='Extract PCA coefficients for each image')
	parser.add_argument('--bases_dir', '-base', 
			action='store', type=str, required=True, 
			help='directory of bases (eigen vectors)')
	parser.add_argument('--exp_id', '-id',
			action='store', type=str, required=True,
			help='experiment id (related to directory where bases and feats are stored)')

	args = parser.parse_args()

	bases_dir = "%s/%s/bases" % (args.bases_dir.strip('/'), args.exp_id)
	bases_pname = "%s/bases.pickle" % bases_dir
	if not os.path.isfile(bases_pname):
		print "%s not exist" % bases_pname
		sys.exit(1)


	with open(bases_pname, "rb") as f:
		W = pickle.load(f) # each col of W is a basis
	D = W.shape[1] # num of bases (feature dimension)
	print "%d NMF bases loaded from %s" % (D, bases_pname)

	sparse = 0
	for w in W.T:
		sparse += sparseness(w)
	
	sparse /= D
	print "Averge sparseness of bases from %s is %.2f" % (bases_dir, sparse)
