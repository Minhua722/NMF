#!/bin/bash

source path.sh

AR="/Users/minhuawu/Documents/JHU/computer_vision/project/database/ar_cropped_data"

num_M=$1
num_W=$2
num_bases=$3
exp_dir=exp_M${num_M}_W${num_W}_B${num_bases}_tst
data_dir=$exp_dir/data

# Prepare train and test set
bash local/ar_data_prep.sh $AR $num_M $num_W $data_dir

# Evaluate face recognition performance based on PCA
python local/ar_compute_pca_bases.py -n $num_bases -in $data_dir -out $exp_dir -id pca
python local/ar_extract_pca_feats.py -in $data_dir -base $exp_dir -id pca
python local/ar_pca_face_recog.py -in $data_dir -out $exp_dir -id pca

# Evaluate face recognition performance based on NMF
# Standard (no constrainti)
for iter in 500; do
	exp_id=nmf_i${iter}
	python local/ar_compute_nmf_bases.py -n $num_bases -in $data_dir -out $exp_dir -id ${exp_id} -i $iter
	python local/ar_extract_nmf_feats.py -in $data_dir -base $exp_dir -id ${exp_id}
	python local/ar_nmf_face_recog.py -in $data_dir -out $exp_dir -id ${exp_id}
done

# Evaluate face recognition performance based on sparse NMF
for sparse in 0.5 0.6 0.7; do
	for iter in 500; do
		exp_id=nmf_W${sparse}_i${iter}
		echo $exp_id
		python local/ar_compute_nmf_bases.py -n $num_bases -in $data_dir -out $exp_dir -id ${exp_id} -i ${iter} -W $sparse
		python local/ar_extract_nmf_feats.py -in $data_dir -base $exp_dir -id ${exp_id}
		python local/ar_nmf_face_recog.py -in $data_dir -out $exp_dir -id ${exp_id}
	done
done


# CNMF
for iter in 500; do
    exp_id=cnmf_i${iter} 
    python local/cnmf.py --dir_name $AR --num_basis $num_bases --num_M $num_M --num_W $num_W --out_dir $exp_dir --alg_id $exp_id --num_iters $iter 
    python local/ar_extract_cnmf_feats.py -in $data_dir -base $exp_dir -id $exp_id
    python local/ar_cnmf_face_recog.py -in $data_dir -out $exp_dir -id $exp_id
    python local/bases_sparseness.py -base $exp_dir -id ${exp_id}
done

# CNMF with sparseness constraint
for sparse in 0.5 0.6 0.7; do
    for iter in 500; do
        exp_id=cnmf_W${sparse}_i${iter}
        python local/cnmf.py --dir_name $AR --num_basis $num_bases --num_M $num_M --num_W $num_W --out_dir $exp_dir --alg_id $exp_id --num_iters $iter -sp $sparse
        python local/ar_extract_cnmf_feats.py -in $data_dir -base $exp_dir -id $exp_id
        python local/ar_cnmf_face_recog.py -in $data_dir -out $exp_dir -id $exp_id
        #python local/bases_sparseness.py -base $exp_dir -id ${exp_id}
    done
done
