#!/bin/bash

source path.sh

ORL="/Users/minhuawu/Documents/JHU/computer_vision/project/database/orl_faces"
exp_dir=exp

python local/test_pca_on_orl.py -in $ORL -out $exp_dir -n 25
python local/test_nmf_on_orl.py -in $ORL -out $exp_dir -i 1000 -n 25
