#!/usr/bin/env python

import numpy as np 
from sklearn.decomposition import PCA
import argparse

import sys, os
from nmf_support import *

if __name__ == '__main__': 

	parser = argparse.ArgumentParser(description='Eigenface Decomposition.')
	parser.add_argument('--num_basis', '-n', 
						action='store', type=int, default=36, 
						help='number of eigenfaces')
	parser.add_argument('--input_path', '-in', 
						action='store', type=str, required=True, 
						help='path to dataset directory')
	parser.add_argument('--output_path', '-out', 
						action='store', type=str, required=True, 
						help='path to output directory')
		
	# args = parser.parse_args('-in ../../cbcl_faces/train/face -out output/test -n 64'.split())
	args = parser.parse_args()
	
	if not os.path.exists(args.input_path):
		print "[ERROR] " + args.input_path + "does not exit!!"
		sys.exit(0)

	if not os.path.exists(args.output_path):
		print "Create output directory " + args.output_path
		os.makedirs(args.output_path)

	V, img_height, img_width = get_V(args.input_path)


	num_basis = args.num_basis
	# if num_basis > V.shape[1]:
	# 	print "Warning: number of eigenfaces cannot be larger than the number of input faces."
	# 	print "Set num_basis = ", V.shape[1]
	# 	num_basis = V.shape[1]
	P = img_height*img_width
	v_bar = np.mean(V, axis=1).reshape(P, 1)
	V = V - v_bar

	pca = PCA(n_components=num_basis)
	
	pca.fit(V.T) # each column corresponding to a face

	W = pca.components_.T

	# print W.shape

	imgs, concat_basis = visualize(W, img_height, img_width, args.output_path, False)	
	mean_face_fname = "%s/mean_face.pgm" % args.output_path
	v_bar = v_bar.astype(np.uint8).reshape(img_height, img_width)
	cv2.imwrite(mean_face_fname, v_bar)

	#cv2.imshow("eigenfaces", concat_basis)
	#cv2.waitKey(0)

	sys.exit(0)



