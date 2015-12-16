#!/usr/bin/env python

import argparse
import os
import sys
import pickle
import numpy as np
import cv2
from nmf_support import *
import Cprojection

def load_data(**kwargs):
    """
    load data to matrix A and X
    **kwargs:
        dir_name: 
            path to directory
        num_W:
            number of woman
        num_M:
            number of man
        num_training:
            number of images from each person
        num_spvs:
            number of images used as labeled
    """
    if not os.path.exists(kwargs['dir_name']):
        print kwargs['dir_name'], 'does not exist!'
        sys.exit(0)
    
    # Read file paths
    paths = []
    labels = []
    spvs = [] 
    for dirname, subdirs, filenames in os.walk(kwargs['dir_name']):
        if len(subdirs) != 0:
            continue
        for fname in filenames:
            img_name, ext = fname.split('.')
            # Choose image files
            if ext in ['jpg', 'png', 'pgm', 'bmp']: 
                gender, person_idx, img_idx = img_name.split('-')
                # Choose training images
                if int(img_idx) in range(1, kwargs['num_training']+1) \
                    or (int(img_idx)-13) in range(1, kwargs['num_training']+1):
                    # Choose man/woman
                    if (gender == 'M' and int(person_idx) in range(kwargs['num_M']+1)) \
                       or (gender == 'W' and int(person_idx) in range(kwargs['num_W']+1)):
                        paths.append(dirname + '/' + fname)
                        labels.append(dirname.split('/')[-1])
                        # Choose whether labled
                        if int(img_idx) in range(1, kwargs['num_spvs']+1) \
                            or (int(img_idx)-13) in range(1, kwargs['num_spvs']+1):
                            spvs.append(True)
                        else:
                            spvs.append(False)
    label_idx = [] # 0-based index
    cnt = 0
    label_dic = {}
    for ll in labels:
        if not label_dic.has_key(ll):
            label_dic[ll] = cnt
            cnt = cnt + 1
        label_idx.append(label_dic[ll])
    
    filelist = zip(paths, label_idx, spvs)
    filelist_labeled = filter(lambda x: x[2], filelist)
    filelist_unlabeled = filter(lambda x: not x[2], filelist)
   
    # for i in xrange(len(paths)):
    #     print filelist[i]
    # sys.exit(0)
    # print filelist_labeled, '\n'
    # print filelist_unlabeled
   
    # Generate matrix X and A
    C = np.zeros((len(filelist_labeled), cnt))
    I = np.identity(len(filelist_unlabeled))
    C = np.pad(C, [(0, 0), (0, len(filelist_unlabeled))], mode='constant')
    I = np.pad(I, [(0, 0), (cnt, 0)], mode='constant')
    A = np.vstack((C, I))

    h, w = cv2.imread(filelist[0][0], 0).shape
    
    # read labeled image
    X = np.empty((w*h, len(filelist)))
    for i in xrange(len(filelist_labeled)):
        pp, ll, _ = filelist_labeled[i] 
        X[:, i] = cv2.imread(pp, 0).flatten()
        A[i, ll] = 1
    bb = len(filelist_labeled)
    # read unlabeled image
    for i in xrange(len(filelist_unlabeled)):
        X[:, bb+i] = cv2.imread(filelist_unlabeled[i][0], 0).flatten()

    # scale each column in X to 0~255
    X = X / X.max(axis=0) * 255
    # print A
    # print X
    return (X, A, (h, w))


def train(A, X, **kwargs):
    """
    CNMF Training
    **kwargs:
        num_basis:
            number of new basises
        num_iters:
            number of iterations
    """
    num_pixels, num_data = X.shape
    A_h, A_w = A.shape
    assert A_h == num_data
    sparse = kwargs['sparse']
    mu = kwargs['mu']    

    Z = np.random.rand(A_w, kwargs['num_basis']) + 1e-3 # random positive initialize
    U = np.random.rand(num_pixels, kwargs['num_basis']) + 1e-3

    if sparse != -1:
        print "project each col of U to be nneg with unchanged L2 norm"
        project_matrix_col(U, sparse, 'unchanged')
    
    for i in xrange(kwargs['num_iters']):
        if (sparse == -1):
            # update U
            U_numerator = X.dot(A).dot(Z)
            U_denominator = U.dot(Z.T).dot(A.T).dot(A).dot(Z)
            U = U * (U_numerator / U_denominator)
        else:
            H = np.dot(Z.T, A.T)
            old_error = obj_error(X, U, H)
            while(1):
                newU = U - mu * (np.dot(U, H) - X).dot(H.T)
                project_matrix_col(newU, sparse, 'unchanged')
                new_error = obj_error(X, newU, H)

                if new_error <= old_error:
                    #print "new error %f" % new_error
                    break

                mu /= 2
                #print "half muW to %f" % muW
                if mu < 1e-10:
                    print "Algorithm converged"
                    return (U, A, Z)

            mu *= 1.2
            U = newU

        # update Z
        Z_numerator = A.T.dot(X.T).dot(U)
        Z_denominator = A.T.dot(A).dot(Z).dot(U.T).dot(U)
        Z = Z * (Z_numerator / Z_denominator)
        # compute objective funciton
        obj_f = ((X - U.dot(Z.T).dot(A.T))**2).sum() / np.prod(X.shape)
        print ' iterations {:d}: {:f}'.format(i+1, obj_f)
    
    return (U, A, Z)

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Constrained Nonnegative Matrix Factorization.')
    parser.add_argument('--num_W', '-W', action='store', type=int, 
                        default=50,
                        help='Number of women.')
    parser.add_argument('--num_M', '-M', action='store', type=int, 
                        default=50,
                        help='Number of men.')
    parser.add_argument('--dir_name', '-d', action='store', type=str, 
                        default='./ar_cropped_data',
                        help='Path to ar_cropped_data dataset.')
    parser.add_argument('--num_training', '-n', action='store', type=int,
                        default=1,
                        help='Number of types used for training.')
    parser.add_argument('--num_spvs', '-r', action='store', type=int,
                        default=1,
                        help='Number of images used as supervised part.')
    parser.add_argument('--num_basis', '-b', action='store', type=int,
                        default=50,
                        help='Number of new basis')
    parser.add_argument('--num_iters', '-i', action='store', type=int,
                        default=500,
                        help='Number of iterations')
    parser.add_argument('--out_dir', '-o', action='store', type=str,
                        default='output',
                        help='Output dir')
    parser.add_argument('--alg_id', action='store', type=str,
                        default='cnmf',
                        help='algorithm id')
    parser.add_argument('--basis_sparse', '-sp', action='store', type=float,
                        default=-1,
                        help='sparseness of the basis (-1 for no sparseness constraint)')
    parser.add_argument('--mu_basis', '-mu', action='store', type=float,
                        default=0.01,
                        help='learning rate (only used under sparseness constraint)') 
	# parser.add_argument('--mask', dest='mask', action='store_true',
    #                     help='Set if use a elipse.')
    # parser.set_defaults(mask=False)
                        
    args = parser.parse_args()
    assert args.num_spvs in range(0, args.num_training + 1), 'num_spvs cannot less than 0, or larger than num_training'
 
    # load the data
    X, A, img_shape = load_data(num_W=args.num_W, 
                                num_M=args.num_M, 
                                dir_name=args.dir_name,
                                num_training=args.num_training,
                                num_spvs=args.num_spvs)

    # train
    U, A, Z = train(A, X, num_basis=args.num_basis, num_iters=args.num_iters, mu=args.mu_basis, sparse=args.basis_sparse)
    
    # visualize
    output_dir = args.out_dir + '/' + args.alg_id + '/bases'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img, concat_basis = visualize(U, img_shape[0], img_shape[1], output_dir, revert=True) 
    
    # cv2.imshow('basis', concat_basis)
    # cv2.waitKey(0)
   
    pickle.dump(U, open(output_dir + '/bases.pickle', 'wb'))
    pickle.dump(A, open(output_dir + '/constraints.pickle', 'wb'))
    pickle.dump(Z, open(output_dir + '/auxiliary.pickle', 'wb'))


