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

-- bunch_size controls the mini-batch for weight updates
local bunch_size    = 128
local weight_decay  = 0.0001
-- replacement controls how many samples will be shown in one epoch
local replacement   = 256
----------------------------------------------------------
-- MLP CONFIGURATION
local mlp_string = "784 inputs 1000 relu 1000 relu 1000 relu 1000 relu 1000 relu 1000 relu 1000 relu 10 log_softmax"
----------------------------------------------------------
 -- data has to be in the same the path where the script is located
local datadir = arg[0]:get_path()
local train_filename = "train-images-idx3-ubyte.mat"
local test_filename  = "t10k-images-idx3-ubyte.mat"
local train_labels_filename = "train-labels-idx1-ubyte.txt"
local test_labels_filename  = "t10k-labels-idx1-ubyte.txt"

-- loads the training and test matrices
print("# Lodaing trainig data...")
local training_samples = matrix.fromFilename(datadir..train_filename)
print("# Lodaing test data...")
local test_samples     = matrix.fromFilename(datadir..test_filename)

-- load training and test labels
local training_labels = matrix.fromTabFilename(datadir..train_labels_filename):scalar_add(1)
local test_labels     = matrix.fromTabFilename(datadir..test_labels_filename):scalar_add(1)

-- the output is an indexed dataset over a identity which allows to produce a
-- local encoding
local identity = dataset.identity(10, 0.0, 1.0)

local function build_input_output_dataset(samples, labels)
  local input_ds = dataset.matrix(samples, {
                                    patternSize = {28, 28},
                                    offset      = {0, 0},
                                    numSteps    = {labels:dim(1), 1},
                                    stepSize    = {28, 28}, })
  local output_ds = dataset.indexed(dataset.matrix(labels), { identity })
  return input_ds, output_ds
end

-- generate training datasets
local train_input_data, train_output_data =
  build_input_output_dataset(training_samples, training_labels)

-- training partition (50000 samples)
local train_input  = dataset.slice(train_input_data,  1, 50000)
local train_output = dataset.slice(train_output_data, 1, 50000)

-- validation partition (10000 samples)
local validation_input  = dataset.slice(train_input_data,  50001, 60000)
local validation_output = dataset.slice(train_output_data, 50001, 60000)

-- generate test dataset
local test_input, test_output =
  build_input_output_dataset(test_samples, test_labels)

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

print("# Training size:   ", train_input:numPatterns())
print("# Validation size: ", validation_input:numPatterns())
print("# Test size:       ", test_input:numPatterns())
print("# Generating MLP")

-- an MLP can be generated given mlp_string to the following function
local thenet = ann.mlp.all_all.generate(mlp_string)
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
for _,B in trainer:iterate_weights("w.*") do B:zeros() end

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
                                                    {28, 28})
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
