#!/usr/bin/env python

#--------------------------------------------------------------------------------
# cmd line examples:
# 	(1) ./factorizer.py -h 									(for help information)
# 	(2) ./factorizer.py -in ../../cbcl_faces/train/face 	(required)
# 						-out output 						(required)
# 						-m 0								(default is 0)
# 						-i 1000 							(default is 2000)
# 						-n 36 								(default is 36)
# 	(3) ./factorizer.py --input_path ../../cbcl_faces/train/face 
# 						--output_path output 
# 						--mode 0
# 						--num_iterations 1000 
# 						--num_basis 36
# 	(2) and (3) is the same
# 	
# For more detail, please use './factorizer.py -h'.
# 	
# Also notice that only mode 0 (i.e. standard nmf) is implemented now. 
# Mode 1, 2, 3 (i.e. nmf with sparseness constraint) will do nothing.
# 	

import cv2
import numpy as np
import argparse
import math

from nmf_support import *

import sys 


if __name__ == '__main__': 

	#------------------------------------------------------
	# Args parser
	#------------------------------------------------------

	parser = argparse.ArgumentParser(description='Do NMF with sparseness constraint.')
	parser.add_argument('--num_basis', '-n', 
						action='store', type=int, default=36, 
						help='number of basis')
	parser.add_argument('--W_sparseness', '-W', 
						action='store', type=float, default=-1,
						help='sparseness (0~1) applied on W (default: -1 for no constraint)')
	parser.add_argument('--H_sparseness', '-H', 
						action='store', type=float, default=-1,
						help='sparseness (0~1) applied on H (default: -1 for no constraint)')
	parser.add_argument('--input_path', '-in', 
						action='store', type=str, required=True, 
						help='path to dataset directory')
	parser.add_argument('--output_path', '-out', 
						action='store', type=str, required=True, 
						help='path to output directory')
	parser.add_argument('--num_iterations', '-i', 
						action='store', type=int, default=2000, 
						help='number of iterations. (default: 2000)')
	
	# args = parser.parse_args('-in ../../cbcl_faces/train/face -out output -i 1000 -n 36 -m 0'.split())
	args = parser.parse_args()

	# Initialize V
	V, img_height, img_width = get_V(args.input_path)

	# Initialize W and H
	W_rows, H_cols = V.shape
	H_rows = W_cols = args.num_basis
	W, H = initial_WH(W_rows, W_cols, H_rows, H_cols)

	# NMF with/without sparse constraint 
	newW, newH = train(V, W, H, args.W_sparseness, args.H_sparseness, args.num_iterations)

	imgs, concat_basis = visualize(newW, img_height, img_width, args.output_path)
	
	#cv2.imshow("basis", concat_basis)
	#cv2.waitKey(0)

	#sys.exit(0)



	

	
	


	

	

	



