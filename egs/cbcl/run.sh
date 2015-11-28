
source path.sh

CBCL="/Users/minhuawu/Documents/JHU/computer_vision/project/database/cbcl_faces"
exp_dir=exp

python local/test_pca_on_cbcl.py -in $CBCL -out $exp_dir -n 49
python local/test_nmf_on_cbcl.py -in $CBCL -out $exp_dir -i 500 -n 49
