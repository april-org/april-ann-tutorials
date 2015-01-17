Multilayer Perceptron - MNIST
=============================

You can download and execute this tutorial executing:

```
$ ./execute.sh
```

It will perform download of APRIL-ANN (if necessary), download of MNIST data to
`tmp/mnist` temporary directory (if necessary), and execution of APRIL-ANN with
script `train.lua`.

Description
-----------

This tutorials is an example of Multilayer Perceptrons using APRIL-ANN
toolkit for basic MNIST task. The following description is intended to describe
the most important parts to new APRIL-ANN users.

It is important to understand the concept of MLPs, you can visit the following
link at [Wikepedia](https://en.wikipedia.org/wiki/Multilayer_perceptron) to
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
-- + 1 is needed because in Lua class indices start at 1
local training_labels = matrix.fromTabFilename(train_labels_path) + 1
local test_labels     = matrix.fromTabFilename(test_labels_path) + 1
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
(not matrices), so the digits shape is lost in the process, but the MLP just
needs that, so it is good.

### Creating the MLP

In APRIL-ANN all ANNs are an instance of what we called components. Components
can be composed in complex structures, and a basic MLP is a stack of several
layers. At the end, the component is wrapped into the trainer abstraction, which
allow to perform automatically a lot of the hard work in training the component.

For MLPs, APRIL-ANN has a simple procedure which receives a string describing
the MLP and returns a component.  **Note** that components have name strings,
allowing to perform look-up of components, and its parameters (weight matrices
and bias vectors) have also names which identified them. For MLPs, the helper
function declares its weights and bias with names `wN` and `bN`, being N the
layer number starting at 1. For this example, we build an MLP with 784 inputs
(28x28), 512 logistic hidden layer and 10 softmax output. **Note** that output
is log-based (`log_softmax`), because we use cross-entropy as loss function and
it requires log-based outputs.

```Lua
local mlp_string = "784 inputs 512 logistic 10 log_softmax"
local thenet = ann.mlp.all_all.generate(mlp_string)
```

It is possible to build more complex neural networks by composing the
components, as for example in
[CNNs tutorial](https://github.com/pakozm/april-ann-tutorials/tree/master/MNIST-CNNs).

### Construction of trainer object and initialization of MLP

Once we have our MLP prepared in a component, variable `thenet`, we wrap it into
a trainer object. The trainer is an abstraction layer which implements several
usual things as weights initialization, dataset traversal for training and/or
validation, and more interesting stuff. So, first, we need to construct the
trainer object by passing it the MLP component, the loss function, the bunch
size (mini-batch size) and the optimization algorithm. The bunch size parameter
allows to train multiple samples at the same time, improving the efficiency of
the system, but reducing its convergence speed. The loss function is used during
training to compute the loss between desired outputs and the MLP outputs. The
optimizer is an algorithm which takes the weight gradients computed by the MLP
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
  name_match = "w.*", -- only initialize weight matrices
  random     =  rnd1,
  inf        = -math.sqrt(6),
  sup        =  math.sqrt(6),
  use_fanin  = true,
  use_fanout = true,
}

-- initializes all biases to zero
for _,b in trainer:iterate_weights("b.*") do b:zeros() end
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
trainer:set_layerwise_option("b.*", "weight_decay", 0)
```

### The pocket algorithm and training loop

We have ready the most important concepts for APRIL-ANN, dataset and trainer
objects. In order to prepare the training loop, it is necessary to prepare three
important tables, the `train_data`, `validation_data` and `test_data` tables.
These three tables will be used with trainer methods `train_dataset`, and
`validate_dataset` to update MLP weights using the given `train_data` table, and
for loss computation (`validate_dataset`) using `validation_data` and
`test_data`.

```Lua
local rnd2 = random(6543) -- for shuffle of the training samples
local rnd3 = random(8527) -- for perturbation of data
-- auxiliary table with the fields necessary for trainer:train_dataset method;
-- it trains selecting a random set of samples with replacement
local train_data = {
  -- the training input is perturbed with gaussian noise
  input_dataset  = dataset.perturbation{ mean=0.0, variance=0.02,
					 random=rnd3,
					 dataset=train_input },
  output_dataset = train_output,
  shuffle        = rnd2, -- indicates to randomly sort data in every epoch
  replacement    = 256,  -- indicates the number of samples in one epoch
}
```

Validation and test data tables are defined to forces the trainer to compute
classification error instead of the default loss function (cross-entropy).
Besides, the bunch size (mini-batch size) is forced to be higher than the
default bunch size value, allowing to perform faster operations with validation
and test data.

``` Lua
-- auxiliary table with the fields necessary for trainer:validate_dataset method
local validation_data = {
  input_dataset  = validation_input,
  output_dataset = validation_output,
  loss = ann.loss.zero_one(), -- force computation of classification error
                              -- instead of the loss given to the trainer
  bunch_size = 512, -- forces a large bunch_size for validation
}

-- auxiliary table with the fields necessary for trainer:validate_dataset method
local test_data = {
  input_dataset  = test_input,
  output_dataset = test_output,
  loss = ann.loss.zero_one(), -- forces computation of classification error
  bunch_size = 512, -- forces a large bunch_size for validation
}
```

The training loop uses an object which implements the known as
[pocket algorithm](https://en.wikipedia.org/wiki/Perceptron#Variants) which
takes note of the best validation loss iteration and keeps a copy of the model.
This copy is updated every time the validation loss is improved, until a simple
convergence criterion is true. Different stopping criteria are available by
default in APRIL-ANN, in this case we use the most basic, we train the MLP until
the validation loss is not improved for 400 epochs. The pocket algorithm is
implemented by the object `trainable.train_holdout_validation`.

```Lua
-- the stopping criterion is 400 epochs without improvement in validation loss
local stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_absolute(400)
local pocket_alg = trainable.train_holdout_validation{
  min_epochs = 100,
  max_epochs = 4000,
  stopping_criterion = stopping_criterion,
}
```

Finally, we use the `pocket_alg` object to control the training loop, and for
every epoch we print data to stdout. The method `pocket_alg:execute()` receives
as argument a function which returns the trainer object and the loss in training
and validation sets. For every epoch which is detected as the best by
`pocket_alg`, the classification error in validation and test is shown at the
screen. For every iteration, a summary string is shown in the screen. This
summary is generated by `pocket_alg` object, and basically contains the
iteration number, the training loss, the validation loss, and the best
validation loss until current iteration.

```Lua
-- train until pocket_alg:execute is false; trian_func uses the given stopping
-- criterion to decide when to return true or false
while pocket_alg:execute(
  function()
    -- update the MLP weights and biases using train_data configuration
    local train_error = trainer:train_dataset(train_data)
    -- computes validation loss
    local val_error   = trainer:validate_dataset(validation_data)
    -- the given closure needs to return the model, the training error and the
    -- validation error
    return trainer, train_error, val_error
end) do
  -- when an epoch is the best, show at screen the validation and test zero-one
  -- errors (classification errors) which is (100 - accuracy)
  if pocket_alg:is_best() then
    local val_rel_error = pocket_alg:get_state_table().validation_error
    local tst_rel_error = trainer:validate_dataset(test_data)
    printf("# VAL  CLASS ERROR %.4f %%  %d\n",
	   val_rel_error*100, val_rel_error*validation_input:numPatterns())
    printf("# TEST CLASS ERROR %.4f %%  %d\n",
	   tst_rel_error*100, tst_rel_error*test_input:numPatterns())
    -- save the input filters (w1 weight matrix)
    local img = ann.connections.input_filters_image(trainer:weights("w1"),
                                                    {28, 28})
    local epoch = pocket_alg:get_state_table().current_epoch
    ImageIO.write(img, string.format("filters-%04d.png", epoch))
  end
  printf("%s\n", pocket_alg:get_state_string())
  io.stdout:flush()
end
```

Once the training has finished, it is possible to take the best model and
serialize it to a file.

```Lua
local best = pocket_alg:get_state_table().best
util.serialize(best, "best_model.lua")
```

At any time, from APRIL-ANN, it is possible to load the serialized model.

```Lua
local best = util.deserialize("best_model.lua")
```

Execution example
-----------------

The execution of `train.lua` script will look as the following example. Note
that the APRIL-ANN disclaimer is only shown when the standard output is a
terminal. If you redirect the output to a file the disclaimer won't be shown.
The following example on a *Intel(R) Core(TM) i5-2320 CPU @ 3.00GHz* (4 cores),
8G of RAM, and APRIL-ANN compiled with Intel MKL has been used, it takes *14
min* and less than *400M* of main memory to work.

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
# Test size:       	10000
# Generating MLP
# Epoch Train-CE Val-ER best_epoch best_val_error 	 time/epoch norm2
# VAL  CLASS ERROR 66.3600 %  6636
# TEST CLASS ERROR 66.0300 %  6603
    1 2.599951 0.663600        1 0.663600 	 cpu: 2.26 wall: 0.75 :: norm2 w=   1.4494  b=   0.0104
# VAL  CLASS ERROR 52.6100 %  5260
# TEST CLASS ERROR 52.7900 %  5278
    2 1.956704 0.526100        2 0.526100 	 cpu: 1.74 wall: 0.68 :: norm2 w=   1.4585  b=   0.0145
# VAL  CLASS ERROR 48.7200 %  4871
# TEST CLASS ERROR 52.0000 %  5199
    3 1.842501 0.487200        3 0.487200 	 cpu: 1.57 wall: 0.65 :: norm2 w=   1.4662  b=   0.0230
    4 1.671368 0.518600        3 0.487200 	 cpu: 1.32 wall: 0.56 :: norm2 w=   1.4876  b=   0.0246
# VAL  CLASS ERROR 27.9100 %  2791
# TEST CLASS ERROR 29.2400 %  2924
    5 1.462207 0.279100        5 0.279100 	 cpu: 1.30 wall: 0.57 :: norm2 w=   1.4925  b=   0.0255
# VAL  CLASS ERROR 27.5300 %  2752
# TEST CLASS ERROR 28.6600 %  2865
    6 1.237785 0.275300        6 0.275300 	 cpu: 1.29 wall: 0.57 :: norm2 w=   1.5056  b=   0.0328
# VAL  CLASS ERROR 23.9700 %  2397
# TEST CLASS ERROR 24.7300 %  2472
    7 1.120062 0.239700        7 0.239700 	 cpu: 1.28 wall: 0.58 :: norm2 w=   1.5162  b=   0.0317
  ...    ...      ...
 2740 0.026637 0.020900     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.8019  b=   0.6560
 2741 0.005630 0.021700     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.8019  b=   0.6579
 2742 0.024891 0.021500     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.8006  b=   0.6613
 2743 0.015984 0.021000     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7951  b=   0.6626
 2744 0.033950 0.021000     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7721  b=   0.6641
 2745 0.022314 0.019900     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7912  b=   0.6622
 2746 0.010544 0.020900     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7914  b=   0.6619
 2747 0.008991 0.019800     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7911  b=   0.6634
 2748 0.008733 0.018900     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7972  b=   0.6626
 2749 0.011232 0.019900     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.7939  b=   0.6645
 2750 0.009123 0.019100     2350 0.018200 	 cpu: 0.61 wall: 0.31 :: norm2 w=   8.8043  b=   0.6613
# Wall total time: 862.101    per epoch: 0.313
# CPU  total time: 1687.841    per epoch: 0.614
# VAL  CLASS ERROR 1.8200 %  182
# TEST CLASS ERROR 1.8900 %  188
```

Data conversion to APRIL-ANN format
-----------------------------------

The `MNIST-utils` directory contains two utilities to transform the MNIST format
into APRIL-ANN matrix format. This conversion is automatically performed by
`execute.sh` script. Particularly, it compiles these two utilities:

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
`MNIST-MLPs` directory, or modify the `datadir` variable into the `train.lua`
script.
