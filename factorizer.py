#!/usr/bin/env python

import cv2
import numpy as np

import time

def get_V(dir_name):
	"""
	Read images in a directory (non-recursively) in to a matrix V. 
	Each column of V represents a image.
	
	Parameters:
		dir_name:	directory name 
	Returns:
		V:	The output matrix
	"""
	
	import glob
	if dir_name[-1] != '/':
		dir_name = dir_name + '/'
		
	paths = glob.glob(dir_name + "*.pgm")

	V_cols = len(paths)
	h, w = cv2.imread(paths[0], 0).shape
	V_rows = h * w
	
	V = np.empty((V_rows, V_cols), dtype=float)

	for i in range(len(paths)):
		img = cv2.imread(paths[i], 0)
		V[:, i] = cv2.imread(paths[i], 0).flatten()
	
	return V

def initial_WH(W_rows, W_cols, H_rows, H_cols):
	"""
	Initialize 2 matrices W and H

	Parameters:
		W_rows, W_cols:	shape of W 
		H_rows, H_cols:	shape of H 
	Returns:
		(W, H):	The output matrix
	"""

	W = np.random.rand(W_rows, W_cols) + 1e-20
	H = np.random.rand(H_rows, H_cols) + 1e-20

	return (W, H)


def train_multiplicative(V, W, H, iternum=5000):

	for i in range(iternum):
		W = W * np.dot(V, H.T) / W.dot(H).dot(H.T)
		H = H * np.dot(W.T, V) / np.dot(W.T, W).dot(H)

		# normalize H and W
		# H_norm = np.sqrt((H*H).sum(axis=1)) c              
		# H = H / H_norm.reshape(H.shape[0], 1)
		# W = W * H_norm.reshape(1, H.shape[0])
		W_norm = np.sqrt((W*W).sum(axis=0))
		W = W / W_norm.reshape(1, W.shape[1])
		H = H * W_norm.reshape(W.shape[1], 1)
		if i % 1000 == 0:
			print i, ":\t", ( (V - np.dot(W, H))*(V - np.dot(W, H)) ).sum() / (V.shape[0] * V.shape[1])

	return (W, H)


def visualize(W):

	for i in range(W.shape[1]):
		img = W[:, i]
		img = img * 255 / img.max()
		img[img>255] = 255
		img = img.astype(np.uint8)

		# print img.dtype
		# print "--------------------\n", W[:, i]
		# time.sleep(20)

		cv2.imwrite("output/" + str(i) + ".pgm", img.reshape((19, 19)))
		# cv2.imshow("xxxx", W[:, i].reshape((19, 19)))
		# cv2.waitKey(0)


if __name__ == '__main__': 

	W_cols = 36
	H_rows = W_cols

	V = get_V("/Users/HU-MENGDIE/CV/final/cbcl_faces/train/face");
	
	W_rows, H_cols = V.shape

	W, H = initial_WH(W_rows, W_cols, H_rows, H_cols)

	newW, newH = train_multiplicative(V, W, H)

	visualize(newW)



