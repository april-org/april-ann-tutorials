MNIST utilities
===============

You can download and convert MNIST data to APRIL-ANN matrix format by executing:

```
$ ./execute.sh
```

Data conversion to APRIL-ANN format
-----------------------------------

This directory contains two utilities to transform the MNIST format into
APRIL-ANN matrix format. You need to compile both utilities:

```
$ g++ -o extract_mnist_images extract_mnist_images.cc
$ g++ -o extract_mnist_labels extract_mnist_labels.cc
```

Once you have them compiled, you can generate the APRIL-ANN matrix format:

```
$ ./extract_mnist_images train-images-idx3-ubyte.gz > train-images-idx3-ubyte.mat
$ ./extract_mnist_labels train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte.txt
$ ./extract_mnist_images t10k-images-idx3-ubyte.gz  > t10k-images-idx3-ubyte.mat
$ ./extract_mnist_labels t10k-labels-idx1-ubyte.gz  > t10k-labels-idx1-ubyte.txt
```
