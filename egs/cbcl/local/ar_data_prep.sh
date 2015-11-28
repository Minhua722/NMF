#!/bin/bash

data_dir=$1 # original directory containing cropped faces
num_M=$2	# number of man used for training
num_W=$3	# number of women used for training
dir=$4		# directory for processed data

mkdir -p $dir
echo -n "" > $dir/train.list
for i in `seq 2 13`; do
	echo -n "" > $dir/test$i.list
done

echo "$num_M men selected for training"
# create train/test filelist for Men
for person in `seq 1 $num_M`; do
	person_id=M-`printf "%03g" $person`
	for photo in 1 14; do
		photo_id=`printf "%02g" $photo` 
		filename=$person_id/$person_id-$photo_id.bmp
		line=`greadlink -f $data_dir/$filename`
		echo -e "$line\t$person_id" >> $dir/train.list
	done

	for i in `seq 2 13`; do
		for photo in $i $[$i+13]; do
			photo_id=`printf "%02g" $photo`
			filename=$person_id/$person_id-$photo_id.bmp
			line=`greadlink -f $data_dir/$filename`
			echo -e "$line\t$person_id" >> $dir/test$i.list
		done
	done
done

echo "$num_M women selected for training"
# create train/test filelist for Men
for person in `seq 1 $num_M`; do
	person_id=W-`printf "%03g" $person`
	for photo in 1 14; do
		photo_id=`printf "%02g" $photo` 
		filename=$person_id/$person_id-$photo_id.bmp
		line=`greadlink -f $data_dir/$filename`
		echo -e "$line\t$person_id" >> $dir/train.list
	done

	for i in `seq 2 13`; do
		for photo in $i $[$i+13]; do
			photo_id=`printf "%02g" $photo`
			filename=$person_id/$person_id-$photo_id.bmp
			line=`greadlink -f $data_dir/$filename`
			echo -e "$line\t$person_id" >> $dir/test$i.list
		done
	done

done

