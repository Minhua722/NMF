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
	
	# extract an elliptical region of each image
	# create the mask
	mask = np.zeros((img_height, img_width))
	cv2.ellipse(mask, center=(img_width/2, img_height/2), \
		axes=(img_width/2,img_height/2+10), \
		angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
	#mask = np.ones((img_height, img_width))*255
	mask_pname = "%s/mask.pickle" % data_dir
	with open(mask_pname, "wb") as f:
		pickle.dump(mask, f)

	V = apply_mask(V_raw, img_height, img_width, mask)
	P = V.shape[0]
	assert(np.where(mask==255)[0].size == P)
	print "Elliptical mask applied on each image"

	v_bar = np.mean(V, axis=1).reshape(P, 1)
	V = V - v_bar
	pca = PCA(n_components=num_basis)
	pca.fit(V.T)
	v_bar = v_bar.astype(np.uint8).reshape(P,)
	mean_face_fname = "%s/mean_face.pgm" % bases_dir
	mean_face = np.zeros((img_height, img_width))
	r_coords, c_coords = np.where(mask == 255)
	mean_face[r_coords, c_coords] = v_bar
	cv2.imwrite(mean_face_fname, mean_face)
	# visualize eigen faces
	W = pca.components_.T
	_, _ = visualize_with_mask(W, img_height, img_width, bases_dir, mask)

	pca_bases_pname = "%s/pca_bases.pickle" % bases_dir
	with open(pca_bases_pname, "wb") as f:
		pickle.dump(W, f)
	print "PCA bases saved in %s" % pca_bases_pname

	mean_face_pname = "%s/mean_face.pickle" % bases_dir
	with open(mean_face_pname, "wb") as f:
		pickle.dump(v_bar, f)
	print "mean faces saved in %s" % mean_face_pname
	

