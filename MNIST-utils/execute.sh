#!/bin/bash
EXECUTE_INIT=$1
script_path=$(dirname $0)
tmp_path=$script_path/../tmp/
base_url=http://yann.lecun.com/exdb/mnist/
train_images_gz=train-images-idx3-ubyte.gz
train_labels_gz=train-labels-idx1-ubyte.gz
test_images_gz=t10k-images-idx3-ubyte.gz
test_labels_gz=t10k-labels-idx1-ubyte.gz
train_images_mat=train-images-idx3-ubyte.mat
train_labels_txt=train-labels-idx1-ubyte.txt
test_images_mat=t10k-images-idx3-ubyte.mat
test_labels_txt=t10k-labels-idx1-ubyte.txt
data_path=$tmp_path/mnist

. $script_path/../configure.sh $tmp_path

mkdir -p $data_path

check_command gunzip g++ ln

install wget

download $base_url $train_images_gz $data_path
download $base_url $train_labels_gz $data_path
download $base_url $test_images_gz $data_path
download $base_url $test_labels_gz $data_path

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

extract_images $data_path/$train_images_gz $data_path/$train_images_mat &&
extract_labels $data_path/$train_labels_gz $data_path/$train_labels_txt &&
extract_images $data_path/$test_images_gz $data_path/$test_images_mat &&
extract_labels $data_path/$test_labels_gz $data_path/$test_labels_txt ||
error "Unable to convert MNIST data to APRIL-ANN matrix format"

ln -fs $data_path/$train_images_mat &&
ln -fs $data_path/$train_labels_txt &&
ln -fs $data_path/$test_images_mat &&
ln -fs $data_path/$test_labels_txt ||
error "Unable to link data and labels"

if [[ -z $EXECUTE_INIT ]]; then
    april-ann init.lua ||
    error "Impossible to execute load script :("
fi
