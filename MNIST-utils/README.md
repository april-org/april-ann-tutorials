MNIST utilities
===============

You can download and convert MNIST data to APRIL-ANN matrix format by executing:

```
$ ./execute.sh
```

MNIST data loading in APRIL-ANN
-------------------------------

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
(not matrices), so the digits shape is lost in the process, however, because the
CNN needs this shape to perform the convolution, the shape would be explicitly
recovered.


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
