Convolutional Neural Networks - MNIST
=====================================

Data conversion to APRIL-ANN format
-----------------------------------

The `utils` directory contains two utilities to transform the MNIST format into
APRIL-ANN matrix format. You need to compile both utilities:

```
$ g++ -o extract_mnist_images extract_mnist_images.cc
$ g++ -o extract_mnist_labels extract_mnist_labels.cc
```

Once you have them compiled, you can generate the APRIL-ANN matrix format:

```
$ ./extract_mnist_images train-images-idx3-ubyte.gz > ../train-images-idx3-ubyte.mat
$ ./extract_mnist_labels train-labels-idx1-ubyte.gz > ../train-labels-idx1-ubyte.txt
$ ./extract_mnist_images t10k-images-idx3-ubyte.gz  > ../t10k-images-idx3-ubyte.mat
$ ./extract_mnist_labels t10k-labels-idx1-ubyte.gz  > ../t10k-labels-idx1-ubyte.txt
```

We use `../` as destination in order to ensure resulting `*.mat` and `*.txt` to
be in the same directory as the `train.lua` script.

Script description
------------------

**Execution example**

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
# Epoch Train-CE Val-CE best_epoch best_val_error    time/epoch    norm2
# CLASS 71.6800 %  7167
# CLASS 70.2700 %  7027
    1 2.281600 2.237152        1 2.237152 	 cpu: 32.18 wall: 8.45 :: norm2 w=   1.4301  b=   0.1147
# CLASS 52.9100 %  5291
# CLASS 51.9700 %  5196
    2 2.180531 2.054351        2 2.054351 	 cpu: 32.09 wall: 8.44 :: norm2 w=   1.4457  b=   0.1582
# CLASS 44.1500 %  4415
# CLASS 43.9800 %  4397
    3 1.894892 1.607224        3 1.607224 	 cpu: 32.43 wall: 8.56 :: norm2 w=   1.6594  b=   0.2886
# CLASS 28.6200 %  2861
# CLASS 28.8600 %  2885
    4 1.401662 1.105189        4 1.105189 	 cpu: 32.11 wall: 8.45 :: norm2 w=   1.8045  b=   0.4935
# CLASS 18.4600 %  1845
# CLASS 19.9700 %  1996
    5 0.974552 0.749916        5 0.749916 	 cpu: 32.00 wall: 8.41 :: norm2 w=   1.8917  b=   0.6344
# CLASS 16.4700 %  1647
# CLASS 17.5100 %  1750
    6 0.716332 0.586917        6 0.586917 	 cpu: 31.88 wall: 8.36 :: norm2 w=   1.9744  b=   0.7432
    ...
    
```
