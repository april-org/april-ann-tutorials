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

mkdir -p $data_path

check_command gunzip g++

install wget

download $base_url $train_images $data_path
download $base_url $train_labels $data_path
download $base_url $test_images $data_path
download $base_url $test_labels $data_path

cd $script_path/../MNIST-utils/ && make && cd - ||
error "Unable to prepare MNIST data"

extract_images_script=$script_path/../MNIST-utils/extract_mnist_images
extract_labels_script=$script_path/../MNIST-utils/extract_mnist_labels

extract_images()
{
    orig=$1
    dest=$2
    if [[ ! -e $dest ]]; then
        message "Extracting images $orig"
        gunzip -d -c $orig | $extract_images_script > $dest
    else
        warning "Skipping extraction of images $orig"
    fi
}

extract_labels()
{
    orig=$1
    dest=$2
    if [[ ! -e $dest ]]; then
        message "Extracting labels $orig"
        gunzip -d -c $orig | $extract_labels_script > $dest
    else
        warning "Skipping extraction of labels $orig"
    fi
}

extract_images $data_path/train-images-idx3-ubyte.gz $data_path/train-images-idx3-ubyte.mat &&
extract_labels $data_path/train-labels-idx1-ubyte.gz $data_path/train-labels-idx1-ubyte.txt &&
extract_images $data_path/t10k-images-idx3-ubyte.gz $data_path/t10k-images-idx3-ubyte.mat &&
extract_labels $data_path/t10k-labels-idx1-ubyte.gz $data_path/t10k-labels-idx1-ubyte.txt ||
error "Unable to convert MNIST data to APRIL-ANN matrix format"
