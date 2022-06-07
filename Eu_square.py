#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:22:47 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def Hamiltonian_Eu(t, k, mu, L_x, L_y, Delta):
    r"""Returns the H_nm matrix for Eu model with:

    .. math::
        
        H = \frac{1}{2}\sum_n^L_x \sum_m^L_y \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            \frac{\Delta}{2i} \tau_x\sigma_z \right] \vec{c}_{n+1,m} +
            \sum_n^{L_x-1} \sum_m^{L_y} \vec{c}^\dagger_{n,m}(-t\tau_z\sigma_0 + i\frac{\Delta}{2}\tau_x\sigma_z)\vec{c}_{n+1,m}
            + H.c.
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    onsite = -mu * np.kron(tau_z, sigma_0)
    hopping_x = -t*np.kron(tau_z, sigma_0) + 1j*Delta/2 * np.kron(tau_x, sigma_z)
    hopping_y = -t*np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_z)
    matrix_diagonal = np.kron(np.eye(L_x*L_y), onsite)     #diagonal part of matrix
    matrix_outside_diagonal_y = np.block([ [np.zeros((4*L_y,4)),np.kron(np.eye(L_y), hopping_y), np.zeros((4*L_y,4*(L_x-1)*L_y-4))],
                                         [np.zeros((4*(L_x-1)*L_y, 4*L_x*L_y))] ])     #upper diagonal part
    matrix = (matrix_diagonal + matrix_outside_diagonal + matrix_outside_diagonal.conj().T)
    return matrix