#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:28:43 2020

@author: ridhhi
"""

#Create fully connected simple neuron network

# 4 inputs and 4 unique weight
# every input has it's own weight
# every neuron has it's one bias
inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2

##Here we conclude that bias not depends on how many of inputs is supplied it's depend on neuron to where it supplied to
##so here we have only one bias because 4 inputs and weights is supplied to one output neuron i.e y=x*w+b


output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+inputs[3]*weights[3]+bias
##output of the neuron network
#print(output)

########################################################################################
########### What if we want to model 3 neuron with 4 inputs ############################
########################################################################################
inputs = [1,2,3,2.5]

weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

# If when we are just modeling single neuron the output is going to be a single value
# In this case we are actually modeling three neurons so output is going to be three value
output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1,
          inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2,
          inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3]

print(output)

# Here we select three neurons with 4 inputs each
# each neuron have a unique set of weights for each unique input
# and each neuron have it's own unique or separate bias