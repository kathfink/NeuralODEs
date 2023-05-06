# -*- coding: utf-8 -*-
"""
Created on Fri May  5 19:07:21 2023

@author: fink4
"""

import torch
import torch.nn as nn

# this is the backbone of the neural ode method
# we define a class called NeuralODE that contains the method for the 
# feedforward neural network

class NeuralODE(nn.Module):
    
    # initialize instance of the class where f is the network
    def __init__(self,func):
        super().__init__()
        self.func = func
    
    # this is the forward pass: ODE initial value problem 
    def forward(self, y0, t, solver):
        soln = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
        soln[0] = y0
        j=1
        
        for t0, t1 in zip(t[:-1],t[1:]):
            dy = solver(self.func, t0, t1-t0, y0)
            y1 = y0 + dy
            soln[j] = y1
            j+=1
            y0=y1
        return soln
