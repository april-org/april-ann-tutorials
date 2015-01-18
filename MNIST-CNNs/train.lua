--[[
  MNIST-CNNs APRIL-ANN tutorial
  Copyright (C) 2014  Francisco Zamora-Martinez

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
]]

----------------------------------------------------------
-- update package.path adding a relative path from current script path
local basedir = arg[0]:get_path()
package.path = package.path .. ";" .. basedir .. "../?/init.lua"
-- Loading data by requiring the module of MNIST-utils directory
local mnist_data = require "MNIST-utils"
-- unpacking data table into local variables
local train_input, train_output,
validation_input, validation_output,
test_input, test_output = table.unpack(mnist_data)
----------------------------------------------------------

-- bunch_size controls the mini-batch for weight updates
local bunch_size    = 128
local weight_decay  = 0.0001
-- replacement controls how many samples will be shown in one epoch
local replacement   = 256
----------------------------------------------------------
-- CONVOLUTION CONFIGURATION
local ishape   = {1, 28, 28}
local conv1    = {1, 5, 5}
local nconv1   = 8
local conv1f   = "relu"
local maxp1    = {1, 2, 2}
local conv2    = {nconv1, 5, 5}
local nconv2   = 16
local conv2f   = "relu"
local maxp2    = {1, 2, 2}
local hidden1  = 128
local hidden1f = "relu"
----------------------------------------------------------

local rnd1 = random(1234)
local rnd2 = random(6543)
local rnd3 = random(8527)

-- auxiliary table with the fields necessary for trainer:train_dataset method;
-- it trains selecting a random set of samples with replacement
local train_data = {
  -- the training input is perturbed with gaussian noise
  input_dataset  = dataset.perturbation{ mean=0.0, variance=0.02,
					 random=rnd3,
					 dataset=train_input },
  output_dataset = train_output,
  shuffle        = rnd2,
  replacement    = replacement,
}

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

print("# Generating MLP")

-- auxiliary function which concatenates a prefix plus a number to generate
-- names automatically
local gname
do
  local d = {} -- dictionary to transform prefix name to index number
  gname = function(prefix)
    local i = (d[prefix] or 0) + 1
    d[prefix] = i
    return prefix .. tostring(i)
  end
end

local function push_convolution(thenet, kernel, n, actf, pooling)
  thenet:
    -- kernel convolution
    push( ann.components.convolution{ kernel=kernel, n=n,
                                      name=gname("conv-w"),
                                      weights=gname("w") } ):
    -- convolution bias
    push( ann.components.convolution_bias{ n=n, ndims=#kernel,
                                           name=gname("conv-b"),
                                           weights=gname("b") } ):
    -- convolution activation function
    push( ann.components.actf[actf]{ name=gname("actf-") } ):
    -- max-pooling
    push( ann.components.max_pooling{ kernel=pooling,
                                      name=gname("pool-") } )
end

-- the net is a stack of components
local thenet = ann.components.stack{ name="stack" }:
  -- the first one transform the dataset output (an array) into a matrix shape
  push( ann.components.rewrap{ size=ishape } )

-- first convolution plus max pooling
push_convolution(thenet, conv1, nconv1, conv1f, maxp1)
-- second convolution plus max pooling
push_convolution(thenet, conv2, nconv2, conv2f, maxp2)

-- the output of the convolution is converted into an array to be the input of
-- a fully connected MLP layer
thenet:push( ann.components.flatten{ name="flatten" } )

-- we compute here the output size of the convolution which will be the input
-- size of the first fully connected layer
local conv_out_size = thenet:precompute_output_size{ 28*28 }[1]

-- first fully connected layer
thenet:push( ann.components.hyperplane{ input=conv_out_size, output=hidden1,
                                        name=gname("hyp-"),
                                        bias_name=gname("b"),
                                        dot_product_name=gname("w"),
                                        bias_weights=gname("b"),
                                        dot_product_weights=gname("w") } ):
  -- activation function
  push( ann.components.actf[hidden1f]{ name=gname("actf-") } ):
  -- dropout to avoid overfitting
  push( ann.components.dropout{ name="dropout", prob=0.5, random=rnd3 } ):
  -- output layer
  push( ann.components.hyperplane{ input=hidden1, output= 10,
                                   name=gname("hyp-"),
                                   bias_name=gname("b"),
                                   dot_product_name=gname("w"),
                                   bias_weights=gname("b"),
                                   dot_product_weights=gname("w") } ):
  -- output activation function
  push( ann.components.actf.log_softmax{ name=gname("actf-") } )

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
-- generates the network and allocates memory for all weight matrices
trainer:build()

-- learning parameters are weight-related, via optimizer (trainer has a wrapper
-- and knows how to set all the options)
trainer:set_option("weight_decay", weight_decay)
-- The bias regularization is a bad thing...
trainer:set_layerwise_option("b.*", "weight_decay", 0)

-- randomize the neural network weights (no biases) in the range
-- [ inf / sqrt(fanin + fanout), sup / sqrt(fanin + fanout) ]
trainer:randomize_weights{
  name_match = "w.*",
  random     =  rnd1,
  inf        = -math.sqrt(6),
  sup        =  math.sqrt(6),
  use_fanin  = true,
  use_fanout = true,
}

-- initializes all biases to zero
for _,b in trainer:iterate_weights("b.*") do b:zeros() end

-- the stopping criterion is 400 epochs without improvement in validation loss
local stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_absolute(400)
local pocket_alg = trainable.train_holdout_validation{
  min_epochs = 100,
  max_epochs = 4000,
  stopping_criterion = stopping_criterion,
}

print("# Epoch Train-CE Val-ER best_epoch best_val_error \t time/epoch norm2")
local cronometro = util.stopwatch()
cronometro:go()

-- train until pocket_alg:execute is false; trian_func uses the given stopping
-- criterion to decide when to return true or false
while pocket_alg:execute(
  function()
    local train_error = trainer:train_dataset(train_data)
    local val_error   = trainer:validate_dataset(validation_data)
    -- the given closure needs to return the model, the training error and the
    -- validation error
    return trainer, train_error, val_error
end) do
  local epoch = pocket_alg:get_state_table().current_epoch
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
                                                    {conv1[2], conv1[3]})
    ImageIO.write(img, string.format("filters-%04d.png", epoch))
  end
  local cpu,wall = cronometro:read()
  printf("%s \t cpu: %.2f wall: %.2f :: norm2 w= %8.4f  b= %8.4f\n",
         pocket_alg:get_state_string(),
  	 cpu/epoch, wall/epoch,
	 trainer:norm2(".*w.*"),
	 trainer:norm2(".*b.*"))
  io.stdout:flush()
end
cronometro:stop()
local cpu,wall = cronometro:read()
local epochs = pocket_alg:get_state_table().current_epoch
printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/epochs)
printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/epochs)

-- take the best model and compute zero-one error (classification error)
local best = pocket_alg:get_state_table().best
local val_rel_error = best:validate_dataset{
  input_dataset = validation_input,
  output_dataset = validation_output,
  loss = ann.loss.zero_one(),
}
local tst_rel_error = best:validate_dataset{
  input_dataset = test_input,
  output_dataset = test_output,
  loss = ann.loss.zero_one(),
}
printf("# VAL  CLASS ERROR %.4f %%  %d\n",
       val_rel_error*100, val_rel_error*validation_input:numPatterns())
printf("# TEST CLASS ERROR %.4f %%  %d\n",
       tst_rel_error*100, tst_rel_error*test_input:numPatterns())
