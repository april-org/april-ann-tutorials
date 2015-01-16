#!/bin/bash
script_path=$(dirname $0)
tmp_path=$script_path/../tmp/
base_url=http://yann.lecun.com/exdb/mnist/
train_images=train-images-idx3-ubyte.gz
train_labels=train-labels-idx1-ubyte.gz
test_images=t10k-images-idx3-ubyte.gz
test_labels=t10k-labels-idx1-ubyte.gz
data_path=$tmp_path/mnist

. $script_path/../configure.sh $tmp_path

check_command ln

$script_path/../MNIST-utils/execute.sh ||
error "Unable to download and prepare MNIST data"

ln -fs $data_path/train-images-idx3-ubyte.mat &&
ln -fs $data_path/train-labels-idx1-ubyte.txt &&
ln -fs $data_path/t10k-images-idx3-ubyte.mat &&
ln -fs $data_path/t10k-labels-idx1-ubyte.txt ||
error "Unable to link data and labels"

april-ann train.lua ||
error "Impossible to execute training script :("
