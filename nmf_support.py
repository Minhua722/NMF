#!/usr/bin/env python

import cv2
import numpy as np
import math 

# import time
# import sys 

def get_V(dir_name):
	"""
	Read images in a directory (non-recursively) in to a matrix V. 
	Each column of V represents a image.
	
	Parameters:
		dir_name:	directory name 
	Returns:
		V:		The output matrix
		h, w:	height and width of each image in the dataset. This
				information is useful when doing visualization
	"""
	
	import glob
	if dir_name[-1] != '/':
		dir_name = dir_name + '/'
		
	paths = glob.glob(dir_name + '*.pgm')

	V_cols = len(paths)
	h, w = cv2.imread(paths[0], 0).shape
	V_rows = h * w
	
	V = np.empty((V_rows, V_cols), dtype=float)

	for i in range(len(paths)):
		img = cv2.imread(paths[i], 0)
		V[:, i] = cv2.imread(paths[i], 0).flatten()
	
	return (V, h, w)

def initial_WH(W_rows, W_cols, H_rows, H_cols):
	"""
	Initialize 2 matrices W and H

	Parameters:
		W_rows, W_cols:	shape of W 
		H_rows, H_cols:	shape of H 
	Returns:
		(W, H):	The output matrices
	"""

	W = np.random.rand(W_rows, W_cols) + 1e-20
	H = np.random.rand(H_rows, H_cols) + 1e-20

	return (W, H)


def train_multiplicative(V, W, H, iternum=2000):
	"""
	Initialize 2 matrices W and H

	Parameters:
		V, W, H:	input matrices 
		iternum:	number of iterations
	Returns:
		(W, H):	The output matrices
	"""

	for i in range(iternum):
		W = W * np.dot(V, H.T) / W.dot(H).dot(H.T)
		H = H * np.dot(W.T, V) / np.dot(W.T, W).dot(H)

		# normalize each column of W and scale H correspondingly
		W_norm = np.sqrt((W*W).sum(axis=0))
		W = W / W_norm.reshape(1, W.shape[1])
		H = H * W_norm.reshape(W.shape[1], 1)
		if i % 500 == 0:
			error = ((V-np.dot(W, H)) * (V-np.dot(W, H))).sum() / (V.shape[0] * V.shape[1])
			print 'average error in iteration %6d is %5.4f' % (i, error)

	return (W, H)


def visualize(W, height, width, path):
	"""
	Visualize matrix W: store each column as a basis image

	Parameters:
		W:		input matrices
		path:	path to output directory
		height:	height of each basis image
		width:	width of each basis image
	Returns:
		imgs:			list of basis images
		concat_basis:	basis images shown in a single mat
	"""
	if path[-1] != '/':
		path = path + '/'

	num_basis = W.shape[1] 
	imgs = []

	#----------------------------------------------------------------
	# Get basis image
	#----------------------------------------------------------------
	for i in range(num_basis):
		img = W[:, i]
		img = img * 255 / img.max()
		img[img>255] = 255
		img = 255 - img.astype(np.uint8).reshape((height, width))

		cv2.imwrite(path + str(i) + '.pgm', img)
		imgs.append(img)

	#---------------------------------------------------------------
	# concatenate basis images to one single image for comparison
	#---------------------------------------------------------------
	ch = int(math.sqrt(num_basis))
	cw = ch
	pad_d = 0

	if ch * cw < num_basis:
		ch = ch + 1
		pad_d = ch * cw - num_basis
	assert ch * cw >= num_basis

	concat_rows = []
	for i in range(ch-1):
		concat_r = reduce(lambda m1, m2: np.hstack((m1, m2)), imgs[i*cw : (i+1)*cw])

		concat_rows.append(concat_r)
	
	concat_r = reduce(lambda m1, m2: np.hstack((m1, m2)), imgs[(ch-1)*cw :])

	if pad_d != 0:
		concat_r = np.pad(concat_r, ((0, 0), (0, pad_d)), mode='constant')
	concat_rows.append(concat_r)

	concat_basis = reduce(lambda m1, m2: np.vstack((m1, m2)), concat_rows)
	cv2.imwrite(path + "basis_imgs.pgm", concat_basis)

	return (imgs, concat_basis)
