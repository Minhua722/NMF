#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import math
import pickle

from sklearn.decomposition import PCA
from nmf_support import *

import sys, os 
import pdb

if __name__ == '__main__': 

	#------------------------------------------------------
	# Args parser
	#------------------------------------------------------

	parser = argparse.ArgumentParser(description='Compute PCA bases on AR training data')
	parser.add_argument('--num_basis', '-n', 
			action='store', type=int, default=36, 
			help='number of basis, default=36')
	parser.add_argument('--input_dir', '-in', 
			action='store', type=str, required=True, 
			help='data dir with a list of image filenames and labels for training')
	parser.add_argument('--output_dir', '-out',
			action='store', type=str, required=True,
			help='directory to store result')
	parser.add_argument('--exp_id', '-id',
			action='store', type=str, required=True,
			help='experiment id (related to directory where bases and feats are stored)')

	args = parser.parse_args()

	num_basis = args.num_basis
	data_dir = args.input_dir.strip('/')
	train_list = "%s/train.list" % data_dir
	if not os.path.isfile(train_list):
		sys.exit(1)

	bases_dir = "%s/%s/bases" % (args.output_dir.strip('/'), args.exp_id)
	if not os.path.exists(bases_dir):
		os.makedirs(bases_dir)

	# load raw images from training list
	V_raw, img_height, img_width, train_labels = load_data(train_list)
	N = len(train_labels)
	assert(V_raw.shape[1] == N)
	print "%d training images of size %dx%d loaded" % (N, img_height, img_width)
	
	V = normalize_data(V_raw)
	P = V.shape[0]

	v_bar = np.mean(V, axis=1).reshape(P, 1)
	V = V - v_bar
	pca = PCA(n_components=num_basis)
	pca.fit(V.T)
	v_bar = v_bar.astype(np.uint8).reshape(P,)
	mean_face_fname = "%s/mean_face.pgm" % bases_dir
	cv2.imwrite(mean_face_fname, v_bar.reshape(img_height, img_width))
	# visualize eigen faces
	W = pca.components_.T
	_, _ = visualize(W, img_height, img_width, bases_dir)

	pca_bases_pname = "%s/pca_bases.pickle" % bases_dir
	with open(pca_bases_pname, "wb") as f:
		pickle.dump(W, f)
	print "PCA bases saved in %s" % pca_bases_pname

	mean_face_pname = "%s/mean_face.pickle" % bases_dir
	with open(mean_face_pname, "wb") as f:
		pickle.dump(v_bar, f)
	print "mean faces saved in %s" % mean_face_pname
	

