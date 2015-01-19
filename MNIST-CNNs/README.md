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

Read the tutorial at
[MNIST-utils](https://github.com/pakozm/april-ann-tutorials/tree/master/MNIST-utils)
to understand how to perform loading of training/test data into APRIL-ANN.

The CNN training script loads this data by requiring directory *MNIST-utils* as
a Lua module, modifying the Lua `package.path` and then calling
`require "MNIST-utils"`.

```Lua
-- update package.path adding a relative path from current script path
local basedir = arg[0]:get_path()
package.path = package.path .. ";" .. basedir .. "../?/init.lua"
-- Loading data by requiring the module of MNIST-utils directory
local mnist_data = require "MNIST-utils"
-- unpacking data table into local variables
local train_input, train_output,
validation_input, validation_output,
test_input, test_output = table.unpack(mnist_data)
```

However, to write the script from *scratch*, it is enough to copy-and-paste the
`MNIST-utils/init.lua` script replacing previous Lua code, and removing the last
return line.

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
                                         weights="w1" } )
-- first convolution bias, over a 3 dimensional matrix with 8 maps
thenet:push( ann.components.convolution_bias{ n=8, ndims=3,
                                              weights="b1" } )
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
                                         weights="w2" } )
-- first convolution bias, over a 3 dimensional matrix with 16 maps
thenet:push( ann.components.convolution_bias{ n=16, ndims=3,
                                              weights="b2" } )
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
                                        bias_weights="b3",
                                        dot_product_weights="w3" } )
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
                                        bias_weights="b4"),
                                        dot_product_weights="w4" } )
-- output activation function
thenet:push( ann.components.actf.log_softmax() )
```

### Construction of trainer object and initialization of CNN

This part is shared with
[MLP tutorial](https://github.com/pakozm/april-ann-tutorials/tree/master/MNIST-MLPs#construction-of-trainer-object-and-initialization-of-mlp).

### The pocket algorithm and training loop

This part is also shared with
[MLP tutorial](https://github.com/pakozm/april-ann-tutorials/tree/master/MNIST-MLPs#the-pocket-algorithm-and-training-loop),
except the training loop core. A minor change is necessary to write
filters to disk correctly because the MLP first layer filters shape is 28x28,
and for the CNN it is the first layer kernel shape, which is 5x5. Just to be
complete, we copied here the whole training loop, which uses the `pocket_alg`
object initialized as described in previous tutorial.

```Lua
-- train until pocket_alg:execute is false; trian_func uses the given stopping
-- criterion to decide when to return true or false
while pocket_alg:execute(
  function()
    -- update the CNN weights and biases using train_data configuration
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
                                                    {5, 5})
    local epoch = pocket_alg:get_state_table().current_epoch
    ImageIO.write(img, string.format("filters-%04d.png", epoch))
  end
  printf("%s\n", pocket_alg:get_state_string())
  io.stdout:flush()
end
```

Execution example
-----------------

The execution of `train.lua` script will look as the following example. Note
that the APRIL-ANN disclaimer is only shown when the standard output is a
terminal. If you redirect the output to a file the disclaimer won't be shown.
Following example on a Intel(R) Core(TM) i5-2320 CPU @ 3.00GHz (4 cores), 8G
of RAM, and APRIL-ANN compiled with Intel MKL, it takes *142 min* and less
than *400M* of main memory.

```
$ april-ann train.lua
APRIL-ANN v0.4.1 COMMIT 2360  Copyright (C) 2012-2015 DSIC-UPV, CEU-UCH
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
# Lodaing trainig data...
# Lodaing test data...
# Training size:        50000
# Validation size:      10000
# Test size:            10000
# Generating MLP
# Epoch Train-CE Val-ER best_epoch best_val_error        time/epoch norm2
# VAL  CLASS ERROR 73.1500 %  7315
# TEST CLASS ERROR 72.0100 %  7200
    1 2.336220 0.731500        1 0.731500        cpu: 31.11 wall: 8.37 :: norm2 w=   1.4756  b=   0.0107
# VAL  CLASS ERROR 55.3000 %  5529
# TEST CLASS ERROR 55.1600 %  5515
    2 2.094389 0.553000        2 0.553000        cpu: 30.99 wall: 8.26 :: norm2 w=   1.4827  b=   0.0235
# VAL  CLASS ERROR 49.1700 %  4916
# TEST CLASS ERROR 49.6700 %  4966
    3 1.946746 0.491700        3 0.491700        cpu: 30.99 wall: 8.33 :: norm2 w=   1.5177  b=   0.0207
    4 1.869048 0.544400        3 0.491700        cpu: 27.40 wall: 7.37 :: norm2 w=   1.5359  b=   0.0336
# VAL  CLASS ERROR 32.0900 %  3208
# TEST CLASS ERROR 32.8600 %  3285
    5 1.579930 0.320900        5 0.320900        cpu: 28.19 wall: 7.57 :: norm2 w=   1.5673  b=   0.0280
# VAL  CLASS ERROR 28.9000 %  2890
# TEST CLASS ERROR 30.3000 %  3030
    6 1.553403 0.289000        6 0.289000        cpu: 28.47 wall: 7.64 :: norm2 w=   1.5828  b=   0.0366
# VAL  CLASS ERROR 27.9600 %  2795
# TEST CLASS ERROR 29.0600 %  2906
    7 1.166165 0.279600        7 0.279600        cpu: 28.74 wall: 7.70 :: norm2 w=   1.6127  b=   0.0372
  ...    ...      ...
 2124 0.008260 0.010300     1728 0.009200        cpu: 14.38 wall: 4.01 :: norm2 w=   3.6466  b=   0.4112
 2125 0.010547 0.010400     1728 0.009200        cpu: 14.38 wall: 4.01 :: norm2 w=   3.6487  b=   0.4136
 2126 0.101701 0.010500     1728 0.009200        cpu: 14.38 wall: 4.01 :: norm2 w=   3.6491  b=   0.4217
 2127 0.052404 0.009700     1728 0.009200        cpu: 14.38 wall: 4.01 :: norm2 w=   3.6496  b=   0.4169
 2128 0.023542 0.010700     1728 0.009200        cpu: 14.38 wall: 4.01 :: norm2 w=   3.6496  b=   0.4168
# Wall total time: 8523.779    per epoch: 4.006
# CPU  total time: 30599.226    per epoch: 14.379
# VAL  CLASS ERROR 0.9200 %  92
# TEST CLASS ERROR 0.9200 %  92
```
