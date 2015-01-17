#!/bin/bash
script_path=$(dirname $0)
tmp_path=$script_path/../tmp/
train_images=train-images-idx3-ubyte.mat
train_labels=train-labels-idx1-ubyte.txt
test_images=t10k-images-idx3-ubyte.mat
test_labels=t10k-labels-idx1-ubyte.txt
data_path=$tmp_path/mnist

. $script_path/../configure.sh $tmp_path

check_command ln

$script_path/../MNIST-utils/execute.sh ||
error "Unable to download and prepare MNIST data"

ln -fs $data_path/$train_images &&
ln -fs $data_path/$train_labels &&
ln -fs $data_path/$test_images &&
ln -fs $data_path/$test_labels ||
error "Unable to link data and labels"

april-ann train.lua ||
error "Impossible to execute training script :("
