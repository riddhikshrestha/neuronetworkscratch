#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:01:41 2020

@author: ridhhi

Batches, Layers and Object:
    
    Why Batches?
  -> Due to parallel operations i.e we can are calculating things in parallel the bigger the batch
  the more parallel operations that we can run, this is why we  tend to do neuron network training 
  on GPUs rather than doing them on CPUs i.e Core Count for CPU is 4-8 but in GPU Core count 100 or 1000 core.
  
  Another advantage is that Batches helps with generalization ie. means it helps us to generalize
  the input sample in batchs

"""
import numpy as np

#inputs = [1,2,3,2.5]    #features from single sample 

inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]   # feature in batches and it's of 3x4

weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]  # Here weights is of 3x4 size

biases = [2,3,0.5]

# To perform dot product of matrix here we have weight of 3x4 size so we need inputs of 4x0 or 4x1 order size
# But here inputs and weights have same shape of 3x4 if we perform dot product we get an error.
# To solve this shape issue we have to use Transpose of matrix: Transpose means swap rows and column
# Here we apply transpose of matrix in weights which will change it's shape to 4x3

outputs = np.dot(inputs,np.array(weights).T) + biases
#print(outputs)


### Now Add Another Layer in above example
inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]   # feature in batches and it's of 3x4

weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]  # Here weights is of 3x4 size

biases = [2,3,0.5]

weights2 = [[0.1,-0.14,0.5],
           [-0.5,0.12,-0.33],
           [-0.44,0.73,-0.13]]  # Here weights is of 3x4 size

biases2 = [-1,2,-0.5]

layer1_output = np.dot(inputs,np.array(weights).T) + biases   # Layer1 Output

layer2_output = np.dot(layer1_output,np.array(weights2).T) + biases2  # Layer2 output

#print(layer2_output)

##
### Here instead of doing above way we here convert the thing in object oriented way
## 
X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]  # 4 features of 3 sample each so it's shape is 4x3 

# Here weights is initialize using random value from -1 to +1
np.random.seed(0)

class Layer_Dense:
    
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)  # Here 0.10 is used to make value range between -0.1 to 0.1
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        
layer1 = Layer_Dense(4,5)  # here 4 is inputs size, we have 4 feature in its each sample in X above. and what ever neurons you want as output neuron, here i used 5
layer2 = Layer_Dense(5,2)  # In layer2 inputs will be output of layer1 i.e 5 neurons so we pass 5 as input of layer2 and output layer of 2 is pass

# Call forward of layer1 object
layer1.forward(X)
#print(layer1.output)

# Call forward of layer2 object
layer2.forward(layer1.output)
print(layer2.output)