#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:35:25 2020
Neural Network from Scratch in Python
Building neural networks in raw Python
@author: ridhhi
"""
import sys
import numpy as np
import matplotlib

#print("Python:",sys.version)
#print("Numpy:",np.__version__)
#print("Matplotlib:",matplotlib.__version__)

#Create fully connected simple neuron network
# 3 inputs and 3 unique weight
# Every unique neuron have a unique bias
inputs = [1.2,5.1,2.1]
weights = [3.1,2.1,8.7]
bias = 3

output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+bias
#output of the neuron network
print(output)

