#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:29:29 2020

@author: ridhhi

Here we are going cleaning code to make more dynamic.

"""
inputs = [1,2,3,2.5]


weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

'''
layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias in zip(weights,biases): # Here zip will list two list, the list of list element wise.
    neuron_output = 0 #Output of given neuron
    for n_input, weight in zip(inputs,neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''

'''
What does bias does in the neuron network?
'''
some_value = -0.5
weight = 0.7
bias = 0.7

#print(some_value*weight)  # we get -0.35 as result
#print(some_value+bias)   # we get 0.1999 as result
# Here we see that bias Offset the value of -ve weight i.e -0.5+0.7 to 0.1999
# At whole time whatever some_value was after bias it stayed a positive value because all it does is offset





'''
Shape: At each dimension what's the size of that dimension

Example:
    l = [2,3,5,7]  ,    Shape = (4,) ,Type: 1D array,vector
    
    An list in python can be an array with numpy and if it is simple list then it is one-dimensional array in numpy
    and in mathematics a list is a Vector.
    
    l1 = [[1,3,6,8],
          [3,5,3,1]]    , Shape = (2,4) ,Type: 2D Array, Matrix
    
    In python we call l1 as list of list. In Numpy this is 2-D array because first list has two lists and each list has 4 element so it Shape is (2,4).
    Array have to be Homologous shape , this means at each dimension they need to have same size for each dimension.
    Matrix is a rectangular array.A list of vector is a Matrix.
    
    l3 = [[[1,2,3,4],
           [3,2,1,5]]
          [[4,3,2,2],
           [6,4,5,3]],
          [[2,8,5,3],
           [1,2,9,4]]]    , Shape = (3,2,4) , Type: 3D array
    
    Here we get 3 list of list,first dimension has 3 elemsnts
    Second dimension we have 2 and Third diminsion we have 4 elements
    
    Tensor: it is an object that can be represented as an array which is used in deep learning where element are used to create tensor for deep learning.
'''

'''
DOT PRODUCT: Element wise multiplication of value of two vectors
    And Dot product of two vectors return a scalar single value.
    Example:
        a = [1,2,3]
        b = [2,3,4]
        dot_prod = a[0]*b[0]+a[1]*b[1]+a[2]b[2] = 20


'''
# Let's implement Our Simple Single Neuron Network using Dot Product
import numpy as np

inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2

output = np.dot(weights,inputs) + bias
#print(output)


# Let 's implement Layer of Neuron Network using Dot product.
inputs = [1,2,3,2.5]


weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

outputs = np.dot(weights,inputs) + biases
print(outputs)