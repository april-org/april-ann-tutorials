#!/bin/bash
script_path=$(dirname $0)
tmp_path=$script_path/../tmp/
train_images=train-images-idx3-ubyte.mat
train_labels=train-labels-idx1-ubyte.txt
test_images=t10k-images-idx3-ubyte.mat
test_labels=t10k-labels-idx1-ubyte.txt
data_path=$tmp_path/mnist

. $script_path/../configure.sh $tmp_path

$script_path/../MNIST-utils/execute.sh 0 ||
error "Unable to download and prepare MNIST data"

april-ann train.lua ||
error "Impossible to execute training script :("
