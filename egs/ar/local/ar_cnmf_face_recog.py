#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import math
import pickle

from sklearn.decomposition import PCA
from nmf_support import *

import sys, os 


if __name__ == '__main__': 

	#------------------------------------------------------
	# Args parser
	#------------------------------------------------------

	parser = argparse.ArgumentParser(description='Face recognition based on CNMF feats')
	parser.add_argument('--input_dir', '-in', 
			action='store', type=str, required=True, 
			help='data dir with nmf feats and labels')
	parser.add_argument('--output_dir', '-out', 
			action='store', type=str, required=True, 
			help='dir for performance report')
	parser.add_argument('--exp_id', '-id',
			action='store', type=str, required=True,
			help='experiment id (related to directory where bases and feats are stored)')

	args = parser.parse_args()

	data_dir = args.input_dir.strip('/')
	feats_dir = "%s/%s" % (args.input_dir.strip('/'), args.exp_id)
	report_dir = "%s/%s" % (args.output_dir.strip('/'), args.exp_id)

	train_feats_pname = "%s/train_feats.pickle" % feats_dir
	train_label_pname = "%s/train_label.pickle" % data_dir

	test_sets = []
	for set_i in range(2, 14):
		test_sets.append("test%d" % set_i)

	with open(train_feats_pname, "rb") as f:
		train_feats = pickle.load(f) # each row is nmf feats for an image
	with open(train_label_pname, "rb") as f:
		train_label = pickle.load(f)

	for set_name in test_sets:
		test_feats_pname = "%s/%s_feats.pickle" % (feats_dir, set_name)
		test_label_pname = "%s/%s_label.pickle" % (data_dir, set_name)
		with open(test_feats_pname, "rb") as f:
			test_feats = pickle.load(f) # each row is nmf feats for an image
		with open(test_label_pname, "rb") as f:
			test_label = pickle.load(f)

		recog_log_fname = "%s/recog_%s.log" % (report_dir, set_name)
		result_fname = "%s/RESULT_%s" % (report_dir, set_name)
		fout_log = open(recog_log_fname, "w")
		fout_res = open(result_fname, "w")

		assert(train_feats.shape[0] == test_feats.shape[0])
		D = train_feats.shape[0]

		Ntest = len(test_label)
		test_count = 0
		gender_correct_count = 0
		face_correct_count = 0
		for t in range(Ntest):
			test_feat = test_feats[:, t].reshape(D, 1)
			ref = test_label[test_count]

			#scores = np.apply_along_axis(np.linalg.norm, 0, train_feats - test_feat)
			#hypo = train_label[np.argmin(scores)]
			# NCC
			hypo =train_label[np.argmax(np.dot(test_feat.T, train_feats))]
			fout_log.write("instance%d\tref: %s\t hypo %s\n" % (test_count, ref, hypo))
			test_count += 1

			ref_gender, _ = ref.split('-')
			hypo_gender, _ = hypo.split('-')

			if (ref_gender == hypo_gender):
				gender_correct_count += 1
			if (ref == hypo):
				face_correct_count += 1

		FAR = float(face_correct_count) / test_count * 100
		GAR = float(gender_correct_count) / test_count * 100
		fout_res.write("Test set:\t %s/test_rec.list\n" % data_dir )
		fout_res.write("===================================================\n")
		fout_res.write("face accuracy rate (FAR%%):\t\t%.2f\t|%d/%d\n" % (FAR, face_correct_count, test_count))
		fout_res.write("gender accuracy rate (GAR%%):\t%.2f\t|%d/%d\n" % (GAR, gender_correct_count, test_count))

		print "Recognition performance reported in %s and %s" % (recog_log_fname, result_fname)	
		fout_log.close()
		fout_res.close()


