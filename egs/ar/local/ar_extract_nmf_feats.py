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
	parser.add_argument('--input_dir', '-in', 
			action='store', type=str, required=True, 
			help='data dir with a list of image filenames and labels for training (extracted features will also be stored here)')

	args = parser.parse_args()

	data_dir = args.input_dir.strip('/')
	train_list = "%s/train.list" % data_dir
	if not os.path.isfile(train_list):
		sys.exit(1)

	test_sets = []
	for set_i in range(2, 14):
		test_sets.append("test%d" % set_i)

	bases_dir = "%s/%s/bases" % (args.bases_dir.strip('/'), args.exp_id)
	bases_pname = "%s/bases.pickle" % bases_dir
	if not os.path.isfile(bases_pname):
		sys.exit(1)

	feats_dir = "%s/%s" % (args.input_dir, args.exp_id)

	with open(bases_pname, "rb") as f:
		W = pickle.load(f) # each col of W is a basis
	D = W.shape[1] # num of bases (feature dimension)
	print "%d NMF bases loaded from %s" % (D, bases_pname)


	##########################################################################
	# Extract training data features
	# load img in each col of V
	V_raw, img_height, img_width, train_labels = load_data(train_list)
	V = normalize_data(V_raw)
	train_label_pname = "%s/train_label.pickle" % data_dir
	with open(train_label_pname, "wb") as f:
		pickle.dump(train_labels, f)
	N = V.shape[1]

	#train_coefs_pname = "%s/coefs.pickle" % bases_dir
	#with open(train_coefs_pname, "rb") as f:
	#	H = pickle.load(f)
	#print H.shape
	#assert(H.shape[0] == D and H.shape[1] == N)

	# mean and variance normailization for each row
	train_feats = np.transpose(np.dot(V.T, W))
	train_feats = train_feats - np.mean(train_feats, axis=0).reshape(1, N)
	train_feats = train_feats / np.std(train_feats, axis=0).reshape(1, N)

	train_feats_pname = "%s/train_feats.pickle" % feats_dir

	with open(train_feats_pname, "wb") as f:
		pickle.dump(train_feats, f)
	#print np.mean(train_feats, axis=0)
	#print np.std(train_feats, axis=0)
	print "train set nmf feats stored in %s" % train_feats_pname

	############################################################################
	# Extract test data features

	for set_name in test_sets:
		test_list = "%s/%s.list" % (data_dir, set_name)
		print "Process %s" % test_list
		# load img in each col of V
		V_raw, img_height, img_width, test_labels = load_data(test_list)
		V = normalize_data(V_raw)
		test_label_pname = "%s/%s_label.pickle" % (data_dir, set_name)
		with open(test_label_pname, "wb") as f:
			pickle.dump(test_labels, f)
		N = V.shape[1]
		print "%d test images of size %dx%d loaded" % (N, img_height, img_width)

		test_feats = np.transpose(np.dot(V.T, W)) # each col is nmf feats for one image
		assert(test_feats.shape[0] == D and test_feats.shape[1] == N)

		# mean and variance normailization for each col
		test_feats = test_feats - np.mean(test_feats, axis=0).reshape(1, N)
		test_feats = test_feats / np.std(test_feats, axis=0).reshape(1, N)

		test_feats_pname = "%s/%s_feats.pickle" % (feats_dir, set_name)
		with open(test_feats_pname, "wb") as f:
			pickle.dump(test_feats, f)
		#print np.mean(test_feats, axis=0)
		#print np.std(test_feats, axis=0)
		print "%s nmf feats stored in %s" % (set_name, test_feats_pname)

