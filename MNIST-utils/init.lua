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

print("# Training size:   ", train_input:numPatterns())
print("# Validation size: ", validation_input:numPatterns())
print("# Test size:       ", test_input:numPatterns())

return { train_input, train_output,
         validation_input, validation_output,
         test_input, test_output }
