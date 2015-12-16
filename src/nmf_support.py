#!/usr/bin/env python

import cv2
import numpy as np
from numpy import linalg as LA
import math

import Cprojection
import sys 

def get_V(dir_name):
	"""
	Read images in a directory (recursively) in to a matrix V. 
	Each column of V represents a image.

	Parameters:
		dir_name:	directory name 
	Returns:
		V:		The output matrix
		h, w:	height and width of each image in the dataset. This
				information is useful when doing visualization
	"""

	# import glob
	# if dir_name[-1] != '/':
	# 	dir_name = dir_name + '/'

	# paths = glob.glob(dir_name + '*.pgm')

	import os

	paths = []
	for dirname, subdirs, filenames in os.walk(dir_name):

		if len(subdirs) != 0:
			continue

		for fname in filenames:
			_, ext = fname.split('.')
			if ext in ['jpg', 'png', 'pgm']:
				paths.append(dirname + '/' + fname)

	# print len(paths)
	# print paths

	V_cols = len(paths)
	h, w = cv2.imread(paths[0], 0).shape
	V_rows = h * w

	V = np.empty((V_rows, V_cols), dtype=float)

	for i in range(len(paths)):
		img = cv2.imread(paths[i], 0)
		V[:, i] = cv2.imread(paths[i], 0).flatten()


	# print V.shape
	# sys.exit(0)

	return (V, h, w)


def load_data(listfile):
	"""
	Read images from a list file.

	Parameters:
		listfile: each line of the list file contains two fields: directory of the image filename, person id
		mask(optional): mask to extract particular region
	Returns:
		V:	The output matrix where each column represents an image
		h, w:	height and width of each image in the dataset.	
		labels:	a list of person id for those images
	"""

	labels = []
	fin = open(listfile, 'r')
	P = 0
	for line in fin:
		items =  line.split()
		labels.append(items[1])
		if P == 0:
			h, w = cv2.imread(items[0], 0).shape
			P = h*w # total number of pixels in each image
			V = np.empty((P, 0), dtype=np.float)
		raw_img = cv2.imread(items[0], 0)
		#raw_img = cv2.equalizeHist(raw_img)
		V = np.append(V, raw_img.reshape(P, 1), axis=1)

	fin.close()
	return (V, h, w, labels)

def apply_mask(V_raw, height, width, mask):
	"""
	Assume each col of V_raw is the loaded rectangular image, apply mask on each to
	extract a particular region (e.g. elliptical)

	Parameters:
		V_raw: matrix of the loaded rectangular images (one for each col)
		height
		width
		mask: (on: 255, off: 1)
	return V_new: each col corresponds to images with only extracted region
	"""

	r_coords, c_coords = np.where(mask==255)
	P = r_coords.size # number of pixel in the active region of the mask
	V_new = np.empty((P, 0), dtype=np.float)
	N = V_raw.shape[1]
	for n in range(N):
		raw_img = V_raw[:,n].T.reshape(height, width)
		region_img = raw_img[r_coords, c_coords]
		V_new = np.append(V_new, region_img.reshape(P, 1), axis=1)	

	return V_new


def normalize_data(V_raw):
	"""
	Normalize each image (one col of V) within range 0~255

	V_raw: matrix for input images
	return V_new: matrix for normalized images
	"""
	
	P = V_raw.shape[0]
	N = V_raw.shape[1]
	V_new = np.empty((P, 0), dtype=np.float)
	for n in range(N):
		raw_img = V_raw[:,n]
		norm_img = raw_img / np.max(raw_img) * 255
		V_new = np.append(V_new, norm_img.reshape(P, 1), axis=1)
	
	return V_new
		

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
	# initialize each row of H to have unit energy
	H_norm = np.sqrt((H*H).sum(axis=1))
	H = H / H_norm.reshape(H.shape[0], 1)
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



def train(V, W, H, W_sparse=0.5, H_sparse=0.5, muW=0.01, muH=0.01, iternum=1000):
	"""
	NMF with sparseness constraint on W

	Parameters:
		V, W, H: 	input matrices
		W_sparse: 	sparseness constraint on W
		mu_W: 		learning rate for W
		iternum: 	number of iterations
	Returns:
		(W, H):	The output matrices

	"""

	dim = W.shape[0]
	num_basis = W.shape[1]

	# if sparseness contraint on W, project each col of W to be nneg, unchanged L2
	if W_sparse != -1:
		assert(W_sparse >= 0 and W_sparse <= 1)
		print "project each col of W to be nneg with unchanged L2 norm"
		project_matrix_col(W, W_sparse, 'unchanged')
	if H_sparse != -1:
		assert(H_sparse >= 0 and H_sparse <= 1)
		project_matrix_row(H, H_sparse, 'unit')
		print "project each row of H to be nneg with unit L2 norm"
		project_matrix_row(H, H_sparse, 'unit')

	for i in range(iternum):

		# update W
		if W_sparse != -1:
			#print "update W"
			old_error = obj_error(V, W, H)
			while(1):

				newW = W - muW * (np.dot(W, H) - V).dot(H.T)
				project_matrix_col(newW, W_sparse, 'unchanged')
				new_error = obj_error(V, newW, H)

				if new_error <= old_error:
					#print "new error %f" % new_error
					break

				muW /= 2
				#print "half muW to %f" % muW
				if muW < 1e-10:
					print "Algorithm converged"
					return (W, H)

			muW *= 1.2
			W = newW
		else:
			#print "update W"
			W = W * np.dot(V, H.T) / W.dot(H).dot(H.T)
			#print "new error %f" % obj_error(V, W, H)

		# update H
		if H_sparse != -1:
			#print "update H"
			old_error = obj_error(V, W, H)
			while(1):
				newH = H - muH * np.dot(W.T, (np.dot(W, H) - V))
				newH_norm = np.sqrt((newH*newH).sum(axis=1))
				newH = newH / newH_norm.reshape(H.shape[0], 1)
				project_matrix_row(newH, H_sparse, 'unit')
				new_error = obj_error(V, W, newH)

				if new_error <= old_error:
					#print "new error %f" % new_error
					break

				muH /= 2
				#print "half muH to %f" % muH
				if muH < 1e-10:
					print "Algorithm converged"
					return (W, H)

			muH *= 1.2
			H = newH
		else:
			#print "update H"
			H = H * np.dot(W.T, V) / np.dot(W.T, W).dot(H)

			# normalize each row of H and scale W correspondingly
			H_norm = np.sqrt((H*H).sum(axis=1))
			H = H / H_norm.reshape(H.shape[0], 1)
			W = W * H_norm.reshape(1, H.shape[0])
			#print "new error %f" % obj_error(V, W, H)

		if (i+1) % 50 == 0:
			print "%d iterations finished, new error: %f" % (i+1, obj_error(V, W, H))

	return (W, H)


def obj_error(V, W, H):
	"""
	Compute approximation mean square error between V and W*H.

	Parameters:
		V, W, H:    input matrices
		error:      average approximation error
	"""

	error = ((V-np.dot(W, H)) * (V-np.dot(W, H))).sum() / (V.shape[0] * V.shape[1])
	return error


def visualize(W, height, width, path, revert=True):
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
		if img.min() < 0: # this if clause is added when visualize eigenfaces
			img = img - img.min()

		img = img * 255 / img.max()
		img[img>255] = 255
		# img[img<0] = 0
		if revert:
			img = 255 - img.astype(np.uint8).reshape((height, width))
		else:
			img = img.astype(np.uint8).reshape((height, width))

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

	# print cw, ch, pad_d
	# print 'num_basis', num_basis

	concat_rows = []
	for i in range(ch-1):
		concat_r = reduce(lambda m1, m2: np.hstack((m1, m2)), imgs[i*cw : (i+1)*cw])
		# print concat_r.shape
		concat_rows.append(concat_r)

	concat_r = reduce(lambda m1, m2: np.hstack((m1, m2)), imgs[(ch-1)*cw :])

	if pad_d != 0:
		concat_r = np.pad(concat_r, ((0, 0), (0, pad_d*width)), mode='constant')
		# print 'last: ', concat_r.shape
		# print pad_d
	concat_rows.append(concat_r)

	# sys.exit(0)

	concat_basis = reduce(lambda m1, m2: np.vstack((m1, m2)), concat_rows)
	cv2.imwrite(path + "basis_imgs.pgm", concat_basis)

	return (imgs, concat_basis)

def visualize_with_mask(W, height, width, path, mask):
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
		region_img = W[:, i]
		if region_img.min() < 0: # this if clause is added when visualize eigenfaces
			region_img = region_img - region_img.min()
		region_img = region_img * 255 / region_img.max()
		region_img[region_img>255] = 255
		region_img = region_img.astype(np.uint8)
		#region_img = 255 - region_img.astype(np.uint8)

		img = np.zeros((height, width))
		r_coords, c_coords = np.where(mask==255)
		assert(region_img.size == r_coords.size)
		img[r_coords, c_coords] = region_img

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

	# print cw, ch, pad_d
	# print 'num_basis', num_basis

	concat_rows = []
	for i in range(ch-1):
		concat_r = reduce(lambda m1, m2: np.hstack((m1, m2)), imgs[i*cw : (i+1)*cw])
		# print concat_r.shape
		concat_rows.append(concat_r)

	concat_r = reduce(lambda m1, m2: np.hstack((m1, m2)), imgs[(ch-1)*cw :])

	if pad_d != 0:
		concat_r = np.pad(concat_r, ((0, 0), (0, pad_d*width)), mode='constant')
		# print 'last: ', concat_r.shape
		# print pad_d
	concat_rows.append(concat_r)

	# sys.exit(0)

	concat_basis = reduce(lambda m1, m2: np.vstack((m1, m2)), concat_rows)
	cv2.imwrite(path + "basis_imgs.pgm", concat_basis)

	return (imgs, concat_basis)

def sparseness(x):
	"""
	Compute sparseness of a vector

	Parameters:	
		x: input vector
	Returns:	
		sparseness
	"""

	dim = len(x)
	L1 = LA.norm(x, 1)
	L2 = LA.norm(x, 2)
	sparseness = (math.sqrt(dim) - L1 / L2) / (math.sqrt(dim) - 1)

	return sparseness


def L1_for_sparseness(D, L2, sparseness):
	"""
	For a vector with dim D, find qualified L1 norm 
	given L2 norm and desired sparseness
	sparseness = (sqrt(n) - L1/L2) / (sqrt(n)-1)
	input args: D -- dimension
				L2 -- specified L2 norm
				sparseness - specified sparseness
	output args: L1 -- qualified L1 norm
	"""

	L1 = (math.sqrt(D) - sparseness * (math.sqrt(D) - 1)) * L2;
	return L1


def project_matrix_col(M, sparse, L2='unchanged'):
	"""
	Project each column of a Matrix. With unchanged/unit L2 norm,
	and L1 norm set to achieve desired sparseness

	Parameters:
		M: 			input matrix
		sparse: 	specified sparseness of each column for projection
		L2: 	indicate whether we want unchanged or unit L2 norm
	"""

	assert(L2 == 'unchanged' or L2 == 'unit')
	dim = M.shape[0]
	COL = M.shape[1]
	for c in range(COL):
		if L2 == 'unchanged':
			target_L2 = LA.norm(M[:, c], 2)
		else:
			target_L2 = 1
		target_L1 = L1_for_sparseness(dim, target_L2, sparse)
		M[:, c] = Cprojection.project_nneg(M[:, c], target_L1, target_L2, False).flatten()

	for c in range(COL):
		assert(abs(sparseness(M[:, c]) - sparse) < 0.1)


def project_matrix_row(M, sparse, L2='unit'):
	"""
	Project each row of a Matrix. With unchanged/unit L2 norm,
	and L1 norm set to achieve desired sparseness

	Parameters:
		M: 			input matrix
		sparse: 	specified sparseness of each column for projection
		L2: 	indicate whether we want unchanged or unit L2 norm
	"""

	assert(L2 == 'unchanged' or L2 == 'unit')
	dim = M.shape[1]
	ROW = M.shape[0]
	for r in range(ROW):
		if L2 == 'unchanged':
			target_L2 = LA.norm(M[r, :], 2)
		else:
			target_L2 = 1
		target_L1 = L1_for_sparseness(dim, target_L2, sparse)
		M[r, :] = Cprojection.project_nneg(M[r, :], target_L1, target_L2, False).flatten()

	for r in range(ROW):

		assert(abs(sparseness(M[r, :]) - sparse) < 0.1)

