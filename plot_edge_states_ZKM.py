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

k = np.pi
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

def edge_states(k, params):
    """ Returns the edge states localized at the edges in a matrix whose columns
     are [left_minus, left_plus, right_minus, right_plus].
    """
    H = Hamiltonian(k=k, **params)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvectors = eigenvectors[:,(2*L-2):(2*L+2)]  # I look only zero energy modes
    states = [0,0,0,0]
    for i in range(4):
        if (np.abs(eigenvectors[0,i]) > 0.01): #If it is a left edge state
            if i<2:
                left_minus = eigenvectors[:4,i]
                states[0] = left_minus
            else:
                left_plus = eigenvectors[:4,i]
                states[1] = left_plus
        else:
            if i<2:
                right_minus = eigenvectors[4*L-4:,i]
                states[2] = right_minus
            else:
                right_plus = eigenvectors[4*L-4:,i]
                states[3] = right_plus
    return np.array(states).T

def spin_structure(k_values, params):
    """
    Returns the angle theta_k and phase phi_nu_k for all values of k.
    """
    theta_k = []
    phi_r_k = []
    phi_l_k = []
    delta_l_k = []
    for k_value in k_values:
        print(k_value)
        states = edge_states(k_value, params)
        right_minus = states[:,2]
        left_minus = states[:,0]
        theta_k.append(2*np.arctan(np.abs(right_minus[0])/np.abs(right_minus[1])))
        phi_r_k.append(np.angle(-right_minus[0]/right_minus[1]))
        phi_l_k.append(np.angle(-left_minus[0]/left_minus[1]))

    return theta_k, phi_r_k, phi_l_k, delta_l_k
#Aligia
t = 1
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

k = 0.8*np.pi
L = 300
theta = 0
n = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, theta=theta)

H = Hamiltonian(k=k, **params)
eigenvalues, eigenvectors = np.linalg.eigh(H)
eigenvectors = eigenvectors[:,(2*L-2):(2*L+2)]
right_minus = eigenvectors[4*L-4:,0]    # depends on parameters
theta_k = 2*np.arctan(np.abs(right_minus[0])/np.abs(right_minus[1]))
print(theta_k)
phi_r_k = np.angle(-right_minus[0]/right_minus[1])
print(phi_r_k)

#states = edge_states(k, params)

k_values = np.linspace(0.8*np.pi, np.pi-0.001)

theta_k, phi_r_k, phi_l_k, delta_l_k = spin_structure(k_values=k_values, params=params)

fig, ax = plt.subplots()
ax.plot(k_values, phi_r_k)
#ax.plot(k_values, phi_l_k)
