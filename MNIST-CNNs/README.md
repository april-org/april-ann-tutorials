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
# VAL  CLASS ERROR 41.5600 %  4156
# TEST CLASS ERROR 42.7700 %  4277
    1 2.195050 0.415600        1 0.415600 	 cpu: 58.10 wall: 19.14 :: norm2 w=   1.4427  b=   0.0510
# VAL  CLASS ERROR 15.1000 %  1509
# TEST CLASS ERROR 15.8900 %  1588
    2 0.804662 0.151000        2 0.151000 	 cpu: 57.60 wall: 19.02 :: norm2 w=   1.4510  b=   0.0578
    3 0.556816 0.163500        2 0.151000 	 cpu: 48.60 wall: 16.05 :: norm2 w=   1.4722  b=   0.0633
# VAL  CLASS ERROR 10.4100 %  1040
# TEST CLASS ERROR 11.7500 %  1174
    4 0.445993 0.104100        4 0.104100 	 cpu: 51.11 wall: 16.89 :: norm2 w=   1.4962  b=   0.0688
# VAL  CLASS ERROR 9.0000 %  900
# TEST CLASS ERROR 9.8900 %  988
    5 0.308289 0.090000        5 0.090000 	 cpu: 52.52 wall: 17.37 :: norm2 w=   1.5174  b=   0.0711
# VAL  CLASS ERROR 8.7000 %  869
# TEST CLASS ERROR 9.5700 %  957
    6 0.283777 0.087000        6 0.087000 	 cpu: 53.49 wall: 17.70 :: norm2 w=   1.5429  b=   0.0760
# VAL  CLASS ERROR 5.9900 %  599
# TEST CLASS ERROR 6.3300 %  632
    7 0.276292 0.059900        7 0.059900 	 cpu: 54.26 wall: 17.99 :: norm2 w=   1.5622  b=   0.0768
  ...    ...      ...
 1398 0.019286 0.011100     1002 0.010500    cpu: 12.46 wall: 3.29 :: norm2 w=   3.0896  b=   1.3822
 1399 0.021655 0.011700     1002 0.010500    cpu: 12.46 wall: 3.29 :: norm2 w=   3.0888  b=   1.3809
 1400 0.018327 0.011600     1002 0.010500    cpu: 12.46 wall: 3.29 :: norm2 w=   3.0901  b=   1.3817
 1401 0.014408 0.011700     1002 0.010500    cpu: 12.46 wall: 3.29 :: norm2 w=   3.0898  b=   1.3805
 1402 0.021803 0.011500     1002 0.010500    cpu: 12.46 wall: 3.29 :: norm2 w=   3.0877  b=   1.3828
# Wall total time: 4617.449    per epoch: 3.293
# CPU  total time: 17465.441    per epoch: 12.458
# CLASS 1.0500 %  104
# CLASS 1.0200 %  102
```
