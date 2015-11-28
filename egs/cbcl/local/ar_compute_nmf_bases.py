#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import math
import pickle

from nmf_support import *

import sys, os 


if __name__ == '__main__': 

	#------------------------------------------------------
	# Args parser
	#------------------------------------------------------

	parser = argparse.ArgumentParser(description='compute NMF based on AR data')
	parser.add_argument('--num_basis', '-n', 
			action='store', type=int, default=36, 
			help='number of basis, default=36')
	parser.add_argument('--W_sparseness', '-W', 
			action='store', type=float, default=-1,
			help='sparseness (0~1) applied on W (default: -1 for no constraint)')
	parser.add_argument('--H_sparseness', '-H', 
			action='store', type=float, default=-1,
			help='sparseness (0~1) applied on H (default: -1 for no constraint)')
	parser.add_argument('--mu_W', '-mW', 
			action='store', type=float, default=0.01,
			help='learining rate for W (default: 0.01 only used under constraint)')
	parser.add_argument('--mu_H', '-mH', 
			action='store', type=float, default=0.01,
			help='learining rate for H (default: 0.01 only used under constraint)')
	parser.add_argument('--input_dir', '-in', 
			action='store', type=str, required=True, 
			help='data dir with a list of image filenames and labels for training')
	parser.add_argument('--output_dir', '-out',
			action='store', type=str, required=True,
			help='directory to store result')
	parser.add_argument('--exp_id', '-id',
			action='store', type=str, required=True,
			help='experiment id (related to directory where bases and feats are stored)')
	parser.add_argument('--num_iterations', '-i', 
			action='store', type=int, default=500, 
			help='number of iterations. (default: 500)')

	args = parser.parse_args()

	data_dir = args.input_dir.strip('/')
	train_list = "%s/train.list" % data_dir
	if not os.path.isfile(train_list):
		sys.exit(1)

	bases_dir = "%s/%s/bases" % (args.output_dir.strip('/'), args.exp_id)
	if not os.path.exists(bases_dir):
		os.makedirs(bases_dir)

	feats_dir = "%s/%s" % (data_dir, args.exp_id)
	if not os.path.exists(feats_dir):
		os.makedirs(feats_dir)

	V_raw, img_height, img_width, train_labels = load_data(train_list)
	print "%d training images of size %dx%d loaded" % (V_raw.shape[1], img_height, img_width)

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

	W_rows, H_cols = V.shape
	H_rows = W_cols = args.num_basis
	print "Initialize basis matrix W (%dx%d) and coefficient matrix (%dx%d)" \
			% (W_rows, W_cols, H_rows, H_cols)
	W, H = initial_WH(W_rows, W_cols, H_rows, H_cols)

	print "Begin to factorize matrix of training data"
	newW, newH = train(V, W, H, args.W_sparseness, args.H_sparseness, \
			args.mu_W, args.mu_H, args.num_iterations)

	_, _ = visualize_with_mask(newW, img_height, img_width, bases_dir, mask)
	bases_pname = "%s/bases.pickle" % bases_dir
	with open(bases_pname, "wb") as f:
		pickle.dump(newW, f)
	print "%d bases saved in %s" % (args.num_basis, bases_dir)

	coefs_pname = "%s/coefs.pickle" % bases_dir
	with open(coefs_pname, "wb") as f:
		pickle.dump(newH, f)
	print "coefficents saved in %s" % coefs_pname


