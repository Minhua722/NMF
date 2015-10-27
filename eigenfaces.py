#!/usr/bin/env python

#--------------------------------------------------------------------------------
# cmd line examples:
# 	./eigenfaces.py -in ../../orl_faces/s1 -out output/eigenfaces -n 36
# 
# --input_path, -in:	input directories (required)
# 						Please put all images into one single directory!
# 						NO SUB-DIRECTORIES !!!
# 						
# --output_path, -out:	output path (required)
# 						Please make sure the directory exists !!!
# 						
# --num_basis, -n:		number of eigenfaces (Optional, default is 16)
# 						

import numpy as np 
from sklearn.decomposition import PCA
import argparse

from nmf_support import *

if __name__ == '__main__': 

	parser = argparse.ArgumentParser(description='Eigenface Decomposition.')
	parser.add_argument('--num_basis', '-n', 
						action='store', type=int, default=16, 
						help='number of eigenfaces')
	parser.add_argument('--input_path', '-in', 
						action='store', type=str, required=True, 
						help='path to dataset directory')
	parser.add_argument('--output_path', '-out', 
						action='store', type=str, required=True, 
						help='path to output directory')
		
	# args = parser.parse_args('-in ../../cbcl_faces/train/face -out output/eigenfaces -n 49'.split())
	args = parser.parse_args()

	V, img_height, img_width = get_V(args.input_path)


	pca = PCA(n_components=args.num_basis)
	
	pca.fit(V.T) # each column corresponding to a face

	W = pca.components_.T

	# print W.shape

	imgs, concat_basis = visualize(W, img_height, img_width, args.output_path)

	cv2.imshow("eigenfaces", concat_basis)
	cv2.waitKey(0)

	sys.exit(0)



