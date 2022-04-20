#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:32:18 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import kwant
import os
import scipy.linalg
from functions import wave_function, Hamiltonian

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def plot_wave_function_real(syst, k, params, n):
    """
    Plot the real part of the nth-wavefunction living in the system.
    """
    ham = syst(k=k, **params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    # Sort according to spin up electrons
    eigenvectors = eigenvectors[::4, :]
    plt.plot(np.real(eigenvectors[:, n])) 
    return eigenvalues[np.argsort(np.abs(eigenvalues))], eigenvectors

def plot_wave_function_imaginary(syst, k, params, n):
    """
    Plot the imaginary part of the nth-wavefunction living in the system.
    """
    ham = syst(k=k, **params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    # Sort according to spin up electrons
    eigenvectors = eigenvectors[::4, :]
    plt.plot(np.imag(eigenvectors[:, n])) 
    return eigenvalues[np.argsort(np.abs(eigenvalues))], eigenvectors

def plot_wave_function(syst, k, params, n):
    """
    Plot the imaginary part of the nth-wavefunction living in the system.
    """
    ham = syst(k=k, **params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    # Sort according to spin up electrons
    #eigenvectors = eigenvectors[::4, :]
    plt.plot(np.abs(eigenvectors[:, n])) 
    return eigenvalues[np.argsort(np.abs(eigenvalues))], eigenvectors


# without crossing 
# t = 1
# Delta_0 = 0.4*t
# Delta_1 = 0.2*t
# mu = t*Delta_0/Delta_1
# lambda_R = 0.5*t

# with crossing
# t = 1
# t_J = t
# mu = t
# Delta_0 = 0.4*t
# Delta_1 = 0.4*t
# lambda_R = 0.5*t

#Aligia
t = 1
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

k = 0.99*np.pi
L = 200
theta = 0
n = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, theta=theta)

#real part of the wavefunction
#fig, ax = plt.subplots(dpi=300)
#eigenvalues, eigenvectors = plot_wave_function_real(Hamiltonian, k=k, params=params, n=n)
#plt.title("Real part of the wavefunction for spin up electron")

#imaginary part of the wavefunction
#fig, ax = plt.subplots(dpi=300)
#eigenvalues, eigenvectors = plot_wave_function_imaginary(Hamiltonian, k=k, params=params, n=n)
#plt.title("Imaginary part of the wavefunction for spin up electron")

#distribution probability
fig, ax = plt.subplots(dpi=300)
eigenvalues, eigenvectors = plot_wave_function(Hamiltonian, k=k, params=params, n=n)
plt.title("Distribution probability for spin up electron")

#%% Spin structure

def spin_structure(k, params):
    """ Returns the phase phi and angle theta for the edge modes.
    """
    H = Hamiltonian(k=k, **params)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Sort according to the absolute values of energy
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))]
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    # Edge right minus
    right_minus= eigenvectors[-4:, 0]
    theta_k = 2*np.arctan(np.abs(right_minus[1]/right_minus[0]))
    return theta_k