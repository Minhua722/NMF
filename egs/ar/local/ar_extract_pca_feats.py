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
	parser.add_argument('--input_dir', '-in', 
			action='store', type=str, required=True, 
			help='data dir with a list of image filenames and labels for training (extracted features will also be stored here)')
	parser.add_argument('--exp_id', '-id',
			action='store', type=str, required=True,
			help='experiment id (related to directory where bases and feats are stored)')

	args = parser.parse_args()

	data_dir = args.input_dir.strip('/')
	train_list = "%s/train.list" % data_dir
	if not os.path.isfile(train_list):
		sys.exit(1)

	test_sets = []
	for set_i in range(2, 14):
		test_sets.append("test%d" % set_i)

	bases_dir = "%s/%s/bases" % (args.bases_dir.strip('/'), args.exp_id)
	pca_bases_pname = "%s/pca_bases.pickle" % bases_dir
	mean_face_pname = "%s/mean_face.pickle" % bases_dir
	
	feats_dir = "%s/%s" % (data_dir, args.exp_id)
	if not os.path.exists(feats_dir):
		os.makedirs(feats_dir)

	with open(pca_bases_pname, "rb") as f:
		W = pickle.load(f) # each col of W is an eigen vector
	D = W.shape[1] # num of bases (feature dimension)
	with open(mean_face_pname, "rb") as f:
		v_bar = pickle.load(f)
	print "%d PCA bases loaded from %s" % (D, pca_bases_pname)
	

	##########################################################################
	# Extract training data features
	# load img in each col of V
	V_raw, img_height, img_width, train_labels = load_data(train_list)
	V = normalize_data(V_raw)
	train_label_pname = "%s/train_label.pickle" % data_dir
	with open(train_label_pname, "wb") as f:
		pickle.dump(train_labels, f)
	P = V.shape[0]
	N = V.shape[1]
	print "%d training images of size %dx%d loaded" % (N, img_height, img_width)
	
	v_bar = v_bar.reshape(P, 1)
	V = V - v_bar
	pca_feats = np.transpose(np.dot(V.T, W)) # each row is pca feats for one image
	assert(pca_feats.shape[0] == D and pca_feats.shape[1] == N)

	# mean and variance normailization for each row
	pca_feats = pca_feats - np.mean(pca_feats, axis=0).reshape(1, N)
	pca_feats = pca_feats / np.std(pca_feats, axis=0).reshape(1, N)
	
	train_pca_feats_pname = "%s/train_pca_feats.pickle" % feats_dir
	with open(train_pca_feats_pname, "wb") as f:
		pickle.dump(pca_feats, f)
	print "train set pca feats stored in %s" % train_pca_feats_pname

	############################################################################
	# Extract test data features
	# load img in each col of V
	for set_name in test_sets:
		test_list = "%s/%s.list" % (data_dir, set_name)
		print "Process %s" % test_list
		V_raw, img_height, img_width, test_labels = load_data(test_list)
		V = normalize_data(V_raw)
		test_label_pname = "%s/%s_label.pickle" % (data_dir, set_name)
		with open(test_label_pname, "wb") as f:
			pickle.dump(test_labels, f)
		N = V.shape[1]
		print "%d test images of size %dx%d loaded" % (N, img_height, img_width)
	
		V = V - v_bar
		pca_feats = np.transpose(np.dot(V.T, W)) # each col is pca feats for one image
		assert(pca_feats.shape[0] == D and pca_feats.shape[1] == N)

		# mean and variance normailization for each col
		pca_feats = pca_feats - np.mean(pca_feats, axis=0).reshape(1, N)
		pca_feats = pca_feats / np.std(pca_feats, axis=0).reshape(1, N)
	
		test_pca_feats_pname = "%s/%s_pca_feats.pickle" % (feats_dir, set_name)
		with open(test_pca_feats_pname, "wb") as f:
			pickle.dump(pca_feats, f)
		print "%s pca feats stored in %s" % (set_name, test_pca_feats_pname)
