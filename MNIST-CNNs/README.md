Convolutional Neural Networks - MNIST
=====================================

You can download and execute this tutorial executing:

```
$ ./execute.sh
```

It will perform download of APRIL-ANN (if necessary), download of MNIST data to
`tmp/mnist` temporary directory (if necessary), and execution of APRIL-ANN with
script `train.lua`.

Description
-----------

This tutorials is an example of Convolutional Neural Networks using APRIL-ANN
toolkit for basic MNIST task. The following description is intended to describe
the most important parts to new APRIL-ANN users.

It is important to understand the concept of CNNs, you can visit the following
link at [deeplearning.net](http://deeplearning.net/tutorial/lenet.html) to
introduce yourself in the topic.

### Data loading

The first step in any train script is to load the data into APRIL-ANN matrices
and data sets. APRIL-ANN implements a cool image library which allow to load
data from raw images, however the MNIST data is stored in its own binary format,
which is transformed into APRIL-ANN ascii matrix format by means of the
utilities in folder `MNIST-utils/`. The data matrix has 28xN rows, where N is
the number of samples, and 28 columns, meaning that matrix data is the
concatenation of all digits by rows. So, for MNIST, we need to load these
matrices and their corresponding training labels. MNIST labels binary files are
transformed into a txt file where every line indicates the class of the
corresponding i-th sample. Training and test data matrices are loaded using
`matrix.fromFilename` loader:

```Lua
-- training and test matrices are loaded given its path
local training_samples = matrix.fromFilename(train_filename_path)
local test_samples     = matrix.fromFilename(test_filename_path)
```

And training and test labels by means of `matrix.fromTabFilename` loader,
which allow to load tabulated ascii files:

```Lua
-- scalar_add(1) is needed because in Lua class indices start at 1
local training_labels = matrix.fromTabFilename(train_labels_path):scalar_add(1)
local test_labels     = matrix.fromTabFilename(test_labels_path):scalar_add(1)
```

APRIL-ANN implements a data set abstraction layer which allow to automate a lot
of the work in ANNs training. For supervised task, like MNIST, it is important
to distinguish the concept of input dataset and output dataset. The input
dataset stores data given as input to the ANN, and the output dataset stores the
data with desired output (or target) values. For MNIST task, the input dataset
is a sliding window traversing data matrix by rows, and the output dataset is a
dictionary which indexes class indices to vectors of length 10 with one
component with 1 and the rest with 0s. This kind of codification is known as
local-encoding, or one-hot vector. The one-hot vector is taken from an identity
dataset (like an identity matrix).

```Lua
-- build the identity dataset for one-hot output vector
local identity = dataset.identity(10, 0.0, 1.0)

-- this auxiliary function takes samples and labels matrices and returns
-- the input and output datasets
local function build_input_output_dataset(samples, labels)
  local input_ds = dataset.matrix(samples, {
                                    patternSize = {28, 28}, -- digit size
                                    -- number of steps in every dimension
                                    numSteps    = {labels:dim(1), 1},
                                    -- step in every dimension
                                    stepSize    = {28, 28}, })
  local output_ds = dataset.indexed(dataset.matrix(labels), { identity })
  return input_ds, output_ds
end

-- generate training datasets
local train_input_data, train_output_data =
  build_input_output_dataset(training_samples, training_labels)

-- generate test dataset
local test_input, test_output =
  build_input_output_dataset(test_samples, test_labels)
```

It is usual in ANNs is to split training data into two sets, train and
validation, using train to adjust model parameters and validation to perform
early stopping and select the best optimization iteration. This split is
performed by means of `dataset.slice` APRIL-ANN object.

```Lua
-- training partition (50000 samples)
local train_input  = dataset.slice(train_input_data,  1, 50000)
local train_output = dataset.slice(train_output_data, 1, 50000)

-- validation partition (10000 samples)
local validation_input  = dataset.slice(train_input_data,  50001, 60000)
local validation_output = dataset.slice(train_output_data, 50001, 60000)
```

So, at this point, all data has been properly loaded in variables *train_input*,
*train_output*, *validation_input* , *validation_output*, *test_input*,
*test_output*. **Note** that all of these datasets produce as output row vectors
(not matrices), so the digits shape is lost in the process, however, because the
CNN needs this shape to perform the convolution, the shape would be explicitly
recovered.

### Creating the CNN layers

In APRIL-ANN all ANNs are an instance of what we called components. Components
can be composed in complex structures, and a basic CNN is a stack of several
layers. At the end, the component is wrapped into the trainer abstraction, which
allow to perform automatically a lot of the hard work in training the component.

So, first we need to instantiate a stack component. **Note** that components
can receive optional name strings, allowing to perform look-up of components.
In this tutorial we will use names only for places where we need them, but
the script `train.lua` has name strings in all the available fields.

```Lua
local thenet = ann.components.stack()
```

The first layer of the CNN is a rewrap layer, which takes as input a vector and
changes its shape, and because `thenet` is a stack, we just pushed the rewrap
component in its top layer. The rewrap would transform the input vector into
a matrix with sizes 1x28x28, being 1 the number of input maps, and 28x28 the
digit shape.

```Lua
thenet:push( ann.components.rewrap{ size={1, 28, 28} } )
```

Following, we need to push the convolution operation. It needs three components,
the convolution, the convolution bias and the activation function. The
convolution bias is given as another component allowing to build CNNs where
convolutional layers have different kind of biases, or non bias at all. The
first convolution takes as input one map of data and uses receptive fields
of 5x5, and produces 8 maps. The activation function is the rectified linear
unit (ReLU), very popular in deep learning community.

```Lua
-- first kernel convolution with receptive fields of 1x5x5
-- and 8 output maps
thenet:push( ann.components.convolution{ kernel={1, 5, 5}, n=8,
                                         weights="W1" } )
-- first convolution bias, over a 3 dimensional matrix with 8 maps
thenet:push( ann.components.convolution_bias{ n=8, ndims=3,
                                              weights="B1" } )
-- first convolution activation function
thenet:push( ann.components.actf.relu() )
```

Following, we push a `max_pooling` component with a kernel of 1x2x2.

```Lua
-- first max-pooling
thenet:push( ann.components.max_pooling{ kernel={1, 2, 2} } )
```

The second convolution has a kernel of 8x5x5 and produces 16 output maps.
It is followed by ReLU activations and a max pooling with 1x2x2 kernel.

```Lua
-- second kernel convolution with receptive fields of 8x5x5
-- and 16 output maps
thenet:push( ann.components.convolution{ kernel={8, 5, 5}, n=16,
                                         weights="W2" } )
-- first convolution bias, over a 3 dimensional matrix with 16 maps
thenet:push( ann.components.convolution_bias{ n=16, ndims=3,
                                              weights="B2" } )
-- first convolution activation function
thenet:push( ann.components.actf.relu() )
-- first max-pooling
thenet:push( ann.components.max_pooling{ kernel={1, 2, 2} } )
```

The output of the convolutional layers needs to be transformed into a row vector
(similar but inverse to the rewrap done at the ANN first layer), because on top
of the convolutional layers we want to put a fully connected hidden layer and a
fully connected output layer. We use the component `flatten` to perform this
operation.

```Lua
thenet:push( ann.components.flatten() )
```

We need to push a fully connected component, called `hyperplane` in APRIL-ANN,
and a hidden ReLU neurons layer. The input of this hidden layer is dependent of
the convolutional layers output, we can use the method `precompute_output_sizes`
to compute a Lua table with the dimension sizes. The hidden layer has 128 units.

```Lua
local conv_out_size = thenet:precompute_output_size{ 28*28 }[1]
-- first fully connected layer
thenet:push( ann.components.hyperplane{ input=conv_out_size, output=128,
                                        bias_weights="B3",
                                        dot_product_weights="W3" } )
-- activation function
thenet:push( ann.components.actf.relu() )
```

Following this hidden layer, to avoid overfitting, we push a
[dropout layer](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf),
which is also very popular in deep learning community. Dropout receives as
parameter a probability value (normally 0.5) and a random numbers generator.

```Lua
local rnd3 = random(8527)
-- dropout component
thenet:push( ann.components.dropout{ prob=0.5, random=rnd3 } ):
```

Finally, we need to push the softmax output layer, which is formed by an
`hyperplane` and a `log_softmax` component. The output of our CNN will be
log-scaled because we will train it using cross-entropy loss function, and in
APRIL-ANN this loss function needs log-scaled values.

```Lua
-- output layer hyperplane
thenet:push( ann.components.hyperplane{ input=128, output= 10,
                                        bias_weights="B4"),
                                        dot_product_weights="W4" } )
-- output activation function
thenet:push( ann.components.actf.log_softmax() )
```

### Construction of trainer object and initialization of CNN

Once we have our CNN prepared in a component, variable `thenet`, we wrap it into
a trainer object. The trainer is an abstraction layer which implements several
usual things as weights initialization, dataset traversal for training and/or
validation, and more interesting stuff. So, first, we need to construct the
trainer object by passing it the CNN component, the loss function, the bunch
size (mini-batch size) and the optimization algorithm. The bunch size parameter
allows to train multiple samples at the same time, improving the efficiency of
the system, but reducing its convergence speed. The loss function is used during
training to compute the loss between desired outputs and the CNN outputs. The
optimizer is an algorithm which takes the weight gradients computed by the CNN
component and update the weight parameters properly. Different optimization
algorithms are implemented in APRIL-ANN, here we use
[ADADELTA](http://arxiv.org/pdf/1212.5701v1.pdf), also a popular algorithm in
deep learning.

```Lua
local bunch_size = 128
-- the trainer knows how ANN components, loss function and optimizer are tightly
-- together
local trainer = trainable.supervised_trainer(thenet,
                                             -- cross entropy for multi-class
                                             -- tasks
                                             ann.loss.multi_class_cross_entropy(),
                                             bunch_size,
                                             -- adadelta is a powerful
                                             -- optimization algorithm
                                             ann.optimizer.adadelta())
```

Once the trainer is constructed, we need to build the ANN component. In this
step, all weight matrices are created, and the input/output requirements of the
components are tested. Following the build step, weights are initialized
randomly in the range `[-sqrt(6/(fanin+fanout)), sqrt(6/fanin+fanout)]`, and
the biases are set to zero.

```Lua
-- generates the network and allocates memory for all weight matrices
trainer:build()

local rnd1 = random(1234) -- for weights initialization
-- randomize the neural network weights (no biases) in the range
-- [ inf / sqrt(fanin + fanout), sup / sqrt(fanin + fanout) ]
trainer:randomize_weights{
  name_match = "W.*", -- only initialize weight matrices
  random     =  rnd1,
  inf        = -math.sqrt(6),
  sup        =  math.sqrt(6),
  use_fanin  = true,
  use_fanout = true,
}

-- initializes all biases to zero
for _,B in trainer:iterate_weights("B.*") do B:zeros() end
```

The ADADELTA algorithm has some options (also known as hyper-parameters) which
can be setup before training. We just modify the weight decay (L2
regularization). The trainer allow to setup the options of the optimizer,
and this options can be setup globally or layerwise. Weight decay will be set
globally to 0.0001 and layerwise 0.0 for biases.

```Lua
-- learning parameters are weight-related, via optimizer (trainer has a wrapper
-- and knows how to set all the options)
trainer:set_option("weight_decay", 0.0001)
-- The bias regularization is a bad thing...
trainer:set_layerwise_option("B.", "weight_decay", 0)
```

### Preparing training loop: pocket algorithm



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

Data conversion to APRIL-ANN format
-----------------------------------

The `MNIST-utils` directory contains two utilities to transform the MNIST format
into APRIL-ANN matrix format. This conversion is automatically performed by
`execute.sh` script. Basically, it doues the compilation of two utilities:

```
$ cd MNIST-utils
$ g++ -o extract_mnist_images extract_mnist_images.cc
$ g++ -o extract_mnist_labels extract_mnist_labels.cc
```

Once they are compiled, it generates the APRIL-ANN matrix format by means of:

```
$ ./extract_mnist_images train-images-idx3-ubyte.gz > train-images-idx3-ubyte.mat
$ ./extract_mnist_labels train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte.txt
$ ./extract_mnist_images t10k-images-idx3-ubyte.gz  > t10k-images-idx3-ubyte.mat
$ ./extract_mnist_labels t10k-labels-idx1-ubyte.gz  > t10k-labels-idx1-ubyte.txt
```

You can convert the data in other directory and put symbolic links in the
`MNIST-CNNs` directory, or modify the `datadir` variable into the `train.lua`
script.
