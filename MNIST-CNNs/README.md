Convolutional Neural Networks - MNIST
=====================================

You can download and execute this tutorial executing:

```
$ ./execute.sh
```

It will perform download of APRIL-ANN (if necessary), download of MNIST data to
`tmp/mnist` temporary directory (if necessary), and execution of APRIL-ANN with
script `train.lua`.

Data conversion to APRIL-ANN format
-----------------------------------

The `MNIST-utils` directory contains two utilities to transform the MNIST format into
APRIL-ANN matrix format. You need to compile both utilities:

```
$ cd MNIST-utils
$ g++ -o extract_mnist_images extract_mnist_images.cc
$ g++ -o extract_mnist_labels extract_mnist_labels.cc
```

Once you have them compiled, you can generate the APRIL-ANN matrix format:

```
$ ./extract_mnist_images train-images-idx3-ubyte.gz > ../MNIST-CNNs/train-images-idx3-ubyte.mat
$ ./extract_mnist_labels train-labels-idx1-ubyte.gz > ../MNIST-CNNs/train-labels-idx1-ubyte.txt
$ ./extract_mnist_images t10k-images-idx3-ubyte.gz  > ../MNIST-CNNs/t10k-images-idx3-ubyte.mat
$ ./extract_mnist_labels t10k-labels-idx1-ubyte.gz  > ../MNIST-CNNs/t10k-labels-idx1-ubyte.txt
```

We use `../MNIST-CNNs/` as destination in order to ensure resulting `*.mat` and
`*.txt` to be in the same directory as the `train.lua` script. You can convert
the data in other directory and put symbolic links in the `MNIST-CNNs`
directory, or modify the `datadir` variable into the `train.lua` script.

Description
-----------



Execution example
-----------------

The execution of `train.lua` script will look as the following example. Note
that the APRIL-ANN disclaimer is only shown when the standard output is a
terminal. If you redirect the output to a file the disclaimer won't be shown.

```
$ april-ann train.lua
APRIL-ANN v0.4.1 COMMIT 2360  Copyright (C) 2012-2015 DSIC-UPV, CEU-UCH
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
# Lodaing trainig data...
# Lodaing test data...
# Training size:   	50000
# Validation size: 	10000
# Generating MLP
# Epoch Train-CE Val-ER best_epoch best_val_error 	 time/epoch norm2
# VAL  CLASS ERROR 73.1500 %  7315
# TEST CLASS ERROR 72.0100 %  7200
    1 2.336220 0.731500        1 0.731500 	 cpu: 40.91 wall: 13.30 :: norm2 w=   1.4756  b=   0.0107
# VAL  CLASS ERROR 55.3000 %  5529
# TEST CLASS ERROR 55.1600 %  5515
    2 2.094389 0.553000        2 0.553000 	 cpu: 40.99 wall: 13.33 :: norm2 w=   1.4827  b=   0.0235
# VAL  CLASS ERROR 49.1700 %  4916
# TEST CLASS ERROR 49.6700 %  4966
    3 1.946746 0.491700        3 0.491700 	 cpu: 41.11 wall: 13.40 :: norm2 w=   1.5177  b=   0.0207
    4 1.869048 0.544400        3 0.491700 	 cpu: 36.06 wall: 11.75 :: norm2 w=   1.5359  b=   0.0336
# VAL  CLASS ERROR 32.0900 %  3208
# TEST CLASS ERROR 32.8600 %  3285
    5 1.579930 0.320900        5 0.320900 	 cpu: 37.01 wall: 12.04 :: norm2 w=   1.5673  b=   0.0280
# VAL  CLASS ERROR 28.9000 %  2890
# TEST CLASS ERROR 30.3000 %  3030
    6 1.553403 0.289000        6 0.289000 	 cpu: 37.50 wall: 12.21 :: norm2 w=   1.5828  b=   0.0366
# VAL  CLASS ERROR 27.9600 %  2795
# TEST CLASS ERROR 29.0600 %  2906
    7 1.166165 0.279600        7 0.279600 	 cpu: 38.03 wall: 12.38 :: norm2 w=   1.6127  b=   0.0372
  ...    ...      ...
 2124 0.008260 0.010300     1728 0.009200 	 cpu: 19.49 wall: 6.30 :: norm2 w=   3.6466  b=   0.4112
 2125 0.010547 0.010400     1728 0.009200 	 cpu: 19.49 wall: 6.30 :: norm2 w=   3.6487  b=   0.4136
 2126 0.101701 0.010500     1728 0.009200 	 cpu: 19.49 wall: 6.30 :: norm2 w=   3.6491  b=   0.4217
 2127 0.052404 0.009700     1728 0.009200 	 cpu: 19.49 wall: 6.30 :: norm2 w=   3.6496  b=   0.4169
 2128 0.023542 0.010700     1728 0.009200 	 cpu: 19.49 wall: 6.30 :: norm2 w=   3.6496  b=   0.4168
# Wall total time: 13410.422    per epoch: 6.302
# CPU  total time: 41465.399    per epoch: 19.486
# VAL  CLASS ERROR 0.9200 %  92
# TEST CLASS ERROR 0.9200 %  92
```
