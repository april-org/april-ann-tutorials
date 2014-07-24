Convolutional Neural Networks - MNIST
=====================================

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

```
$ april-ann train.lua
APRIL-ANN v0.3.1 COMMIT 1689  Copyright (C) 2012-2014 DSIC-UPV, CEU-UCH
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
# Lodaing trainig data...
# Lodaing test data...
# Training size:   	50000
# Validation size: 	10000
# Generating MLP
# Epoch Train-CE Val-ER best_epoch best_val_error    time/epoch    norm2
# CLASS 71.6800 %  7167
# CLASS 70.2700 %  7027
    1 2.281600 0.716800        1 0.716800 	 cpu: 22.44 wall: 6.17 :: norm2 w=   1.4301  b=   0.1147
# CLASS 52.9100 %  5291
# CLASS 51.9700 %  5196
    2 2.180531 0.529100        2 0.529100 	 cpu: 22.15 wall: 5.95 :: norm2 w=   1.4457  b=   0.1582
# CLASS 44.1500 %  4415
# CLASS 43.9800 %  4397
    3 1.894892 0.441500        3 0.441500 	 cpu: 22.17 wall: 5.93 :: norm2 w=   1.6594  b=   0.2886
# CLASS 28.6200 %  2861
# CLASS 28.8600 %  2885
    4 1.401662 0.286200        4 0.286200 	 cpu: 22.02 wall: 5.85 :: norm2 w=   1.8045  b=   0.4935
# CLASS 18.4600 %  1845
# CLASS 19.9700 %  1996
    5 0.974552 0.184600        5 0.184600 	 cpu: 21.88 wall: 5.79 :: norm2 w=   1.8917  b=   0.6344
# CLASS 16.4700 %  1647
# CLASS 17.5100 %  1750
    6 0.716332 0.164700        6 0.164700 	 cpu: 21.72 wall: 5.73 :: norm2 w=   1.9744  b=   0.7432
# CLASS 14.9800 %  1498
# CLASS 15.5300 %  1553
    7 0.614467 0.149800        7 0.149800 	 cpu: 21.68 wall: 5.71 :: norm2 w=   2.0084  b=   0.7744
    ...
```
