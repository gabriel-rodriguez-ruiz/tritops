# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:11:47 2022

@author: gabri
"""
import sympy as sp
from sympy.physics.quantum import TensorProduct

phi = sp.symbols("phi", real=True)
t_1 = sp.symbols("t_1", real=True)
t_2 = sp.symbols("t_2", real=True)
epsilon = sp.symbols("epsilon", real=True)
theta = sp.symbols("theta", real=True)

sigma_0 = sp.Matrix([[1, 0],
                   [0, 1]])
sigma_x = sp.Matrix([[0, 1],
                   [1, 0]])
sigma_y = sp.Matrix([[0, -1j],
                   [1j, 0]])
sigma_z = sp.Matrix([[1, 0],
                   [0, -1]])
tau_0 = sp.Matrix([[1, 0],
                   [0, 1]])
tau_x = sp.Matrix([[0, 1],
                   [1, 0]])
tau_y = sp.Matrix([[0, -1j],
                   [1j, 0]])
tau_z = sp.Matrix([[1, 0],
                   [0, -1]])
                  
H = ( sp.cos(phi/2)* (t_1*TensorProduct(tau_x, sigma_0) +
                     t_2*TensorProduct(tau_y, sigma_x)) +
     epsilon*TensorProduct(tau_0, sigma_z))
eigenvals = list(H.eigenvals().keys())
# for i, eigenval in enumerate(eigenvals):
#     eigenval = eigenval.subs(eigenval.args[0], eigenval.args[0].nsimplify([sp.sqrt(2)]))
#     eigenval = eigenval.subs(eigenval.args[1], eigenval.args[1].nsimplify([1/2]))
#     eigenval = eigenval.args[1].factor()
#     eigenvals[i] = eigenval
    # print(eigenval.args)
    
eigenval = eigenvals[0]

# sp.diff(eigenvals[0], phi)
