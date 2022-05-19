# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:11:47 2022

@author: gabri
"""
import sympy as sp
from sympy.physics.quantum import TensorProduct

phi = sp.symbols("phi")
t_1 = sp.symbols("t_1")
t_2 = sp.symbols("t_2")
epsilon = sp.symbols("epsilon")
tau_x = sp.Matrix([[0, 1],
                   [1, 0]])
sigma_x = sp.Matrix([[0, 1],
                   [1, 0]])
sigma_z = sp.Matrix([[1, 0],
                   [0, -1]])
tau_0 = sp.Matrix([[1, 0],
                   [0, 1]])
tau_x = sp.Matrix([[0, 1],
                   [1, 0]])
tau_y = sp.Matrix([[0, -1j],
                   [1j, 0]])
sigma_0 = sp.Matrix([[1, 0],
                   [0, 1]])
                  
H = ( sp.cos(phi/2) * (t_1*TensorProduct(tau_x, sigma_0) +
                     t_2*TensorProduct(tau_y, sigma_x)) +
                    epsilon*TensorProduct(tau_0, sigma_z) +
                    sp.sin(phi/2)*(t_2*TensorProduct(tau_x, sigma_x) -TensorProduct(tau_y, sigma_x)))

eigenvals = list(H.eigenvals().keys())
sp.diff(eigenvals[0], phi)
