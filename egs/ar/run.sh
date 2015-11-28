#!/bin/bash

source path.sh

AR="/Users/minhuawu/Documents/JHU/computer_vision/project/database/ar_cropped_data"

num_M=50
num_W=50
num_bases=50
exp_dir=exp_M${num_M}_W${num_W}
data_dir=$exp_dir/data

# Prepare train and test set
bash local/ar_data_prep.sh $AR $num_M $num_W $data_dir

# Evaluate face recognition performance based on PCA
python local/ar_compute_pca_bases.py -n $num_bases -in $data_dir -out $exp_dir -id pca
python local/ar_extract_pca_feats.py -in $data_dir -base $exp_dir -id pca
python local/ar_pca_face_recog.py -in $data_dir -out $exp_dir -id pca

# Evaluate face recognition performance based on NMF
# Standard (no constraint)
python local/ar_compute_nmf_bases.py -n $num_bases -in $data_dir -out $exp_dir -id nmf -i 1000
python local/ar_extract_nmf_feats.py -in $data_dir -base $exp_dir -id nmf
python local/ar_nmf_face_recog.py -in $data_dir -out $exp_dir -id nmf
