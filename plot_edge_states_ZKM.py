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
t = 1
t_J = t
mu = t
Delta_0 = 0.4*t
Delta_1 = 0.4*t
lambda_R = 0.5*t

#Aligia
# t = 1
# mu = 2*t
# Delta_0 = 4*t
# Delta_1 = 2.2*t
# lambda_R = 7*t

k = np.pi-0.01*np.pi
L = 400
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
    delta_r_k = []
    for k_value in k_values:
        print(k_value)
        states = edge_states(k_value, params)
        right_minus = states[:,2]
        right_plus = states[:, 3]
        left_minus = states[:,0]
        left_plus = states[:, 1]
        theta_k.append(2*np.arctan(np.abs(right_minus[0])/np.abs(right_minus[1])))
        phi_r_k.append(np.angle(-right_minus[0]/right_minus[1]))
        phi_l_k.append(np.angle(-left_minus[0]/left_minus[1]))
        delta_l_k.append(-1/2*np.angle(left_plus[0]/left_minus[1]))
        delta_r_k.append(-1/2*np.angle(right_plus[0]/right_minus[1]))

    return theta_k, phi_r_k, phi_l_k, delta_l_k, delta_r_k
#Aligia
t = 1
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

k = 999/1000*np.pi
L = 200
theta = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, theta=theta)

states = edge_states(k, params)

k_values = np.linspace(0.99*np.pi, np.pi-0.001)

theta_k, phi_r_k, phi_l_k, delta_l_k, delta_r_k = spin_structure(k_values=k_values, params=params)

fig, ax = plt.subplots()
ax.plot(k_values, phi_l_k, label=r"$\varphi_{l,k}$")
ax.plot(k_values, phi_r_k, label=r"$\varphi_{r,k}$")
ax.plot(k_values, delta_l_k, "*", label=r"$\delta_{l,k}$")
ax.plot(k_values, delta_r_k, "*", label=r"$\delta_{r,k}$")
ax.plot(k_values, theta_k, label=r"$\theta_k$")
ax.legend()

#%% Z = rho_k*e^iphi
# Numerical calculation of Z
from functions import spectrum, Hamiltonian
#Aligia
t = 1
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

L = 200
theta = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, theta=theta)

k = np.linspace(0, np.pi, 150)
spectrum = spectrum(Hamiltonian, k, **params)

fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
# fig.set_figwidth(246/72)    # in inches, \columnwith=246pt and 1pt=1/72 inch
ax.plot(
    k, spectrum, linewidth=0.1, color="m"
)  # each column in spectrum is a separate dataset
ax.plot(
    k, spectrum[:, 398:402], linewidth=1, color="c"
)  # each column in spectrum is a separate dataset



#%% Plot of edge states for definite k

# without crossing 
t = 1
t_J = t/2
Delta_0 = 0.4*t
Delta_1 = 0.2*t
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t

# k = np.pi/2
k = np.pi-0.5*np.pi
L = 300
theta = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, theta=theta)

def zero_energy_states(k, params):
    """ Returns the edge states in the following order
      [left_minus, left_plus, right_minus, right_plus].
    """
    H = Hamiltonian(k=k, **params)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvectors = eigenvectors[:,(2*L-2):(2*L+2)]
    left_minus = np.zeros(len(eigenvectors))
    left_plus = np.zeros(len(eigenvectors))
    right_minus = np.zeros(len(eigenvectors))
    right_plus = np.zeros(len(eigenvectors))
    for i in range(4):
        if (np.abs(eigenvectors[0,i]) > 0.01): #If it is a left edge state
            if i<2:
                left_minus = eigenvectors[:,i]
            else:
                left_plus = eigenvectors[:,i]
        else:
            if i<2:
                right_minus = eigenvectors[:,i]
            else:
                right_plus = eigenvectors[:,i]
    return left_minus, left_plus, right_minus, right_plus


fig, ax = plt.subplots()
ax.plot(np.abs(zero_energy_states(k, params)[0][::4]))

#%%
plt.close
plt.figure()

#Aligia
t = 1
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

# without crossing 
# t = 1
# t_J = t/2
# Delta_0 = 0.4*t
# Delta_1 = 0.2*t
# mu = t*Delta_0/Delta_1
# lambda_R = 0.5*t

L = 400
theta = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, theta=theta)

k_values = np.linspace(0, np.pi-0.01*np.pi, 100)
rho_k = []
phi_k = []
for k in k_values:
    left_minus = zero_energy_states(k, params)[2]
    Z = np.sum((left_minus[::4]/np.linalg.norm(left_minus[::4]))**2)
    #Z = np.sum((eigenvectors[::4,0]/np.linalg.norm(eigenvectors[::4,0]))**2)
    rho_k.append(np.abs(Z))
    phi_k.append(np.angle(Z))
    
# print(rho_k)
# print(phi_k)
plt.plot(k_values, rho_k)
#plt.plot(k_values, rho_k)