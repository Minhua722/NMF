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
	parser.add_argument('--mode', '-m' , 
						action='store', type=int, default=0, choices=range(0, 4), 
						help='0 - non constraint; 1 - constraint on W; 2 - constraint on W; \
								3 - constraint on both. (default: 0)')
	parser.add_argument('--num_basis', '-n', 
						action='store', type=int, default=36, 
						help='number of basis')
	parser.add_argument('--H_saprseness', '-H', 
						action='store', type=float, default=0.5,
						help='sparseness applied on H (default: 0.5)')
	parser.add_argument('--W_saprseness', '-W', 
						action='store', type=float, default=0.5,
						help='sparseness applied on W (default: 0.5)')
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

	# Standard nmf without constraint
	if args.mode is 0:
		print '\nMultiplicative update without any constraint'
		print 'shape of V: %d, %d' % V.shape
		print 'shape of W: %d, %d' % W.shape
		print 'shape of H: %d, %d\n' % H.shape
		newW, newH = train_multiplicative(V, W, H, args.num_iterations)

	elif args.mode is 1:
		print '\nMultiplicative update with sparseness constraint on W'
		print '!!!! Not implemented yet'
		sys.exit(0)

	elif args.mode is 2:
		print '\nMultiplicative update with sparseness constraint on H'
		print '!!!! Not implemented yet'
		sys.exit(0)
		
	else:
		print '\nMultiplicative update with sparseness constraint on both W and H\n'
		print '!!!! Not implemented yet'
		sys.exit(0)


	imgs, concat_basis = visualize(newW, img_height, img_width, args.output_path)
	
	cv2.imshow("basis", concat_basis)
	cv2.waitKey(0)

	sys.exit(0)



	

	
	


	

	

	



