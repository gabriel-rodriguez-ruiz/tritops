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

directory = os.getcwd()
path, file = os.path.split(directory)

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def onsite_ZKM(site, mu, Delta_0, Delta_1, lambda_R, k, t):
    return ( (-2*t*np.cos(k) - mu) * np.kron(tau_z, np.eye(2)) +
            (Delta_0 + 2*Delta_1*np.cos(k)) * np.kron(tau_x, np.eye(2))  +
            2*lambda_R*np.sin(k) * np.kron(tau_z, sigma_x) 
             )

def hopping_ZKM(site1, site2, t, Delta_0, Delta_1,lambda_R, k):
        return ( -t * np.kron(tau_z, np.eye(2)) +
                1j*lambda_R * np.kron(tau_z, sigma_z) +
                Delta_1 * np.kron(tau_x, np.eye(2)))

def plot_density(syst, params, n):
    """
    Plot the nth-density probability living in the system.
    """
    density = kwant.operator.Density(syst)
    ham = syst.hamiltonian_submatrix(params=params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    plt.plot(density(eigenvectors[:, n]))   
    kwant.plotter.plot(syst, site_color=density(eigenvectors[:, n]), site_size=0.5,
                       cmap='gist_heat_r')
    return eigenvectors[:, n]
    
def make_chain_finite(t=1, mu=1, Delta_0=1, Delta_1=1, lambda_R=0.5, L=25, k=0, theta=0):
    """
    Create a finite chain.
    """
    chain_finite = kwant.Builder()
    lat = kwant.lattice.chain(norbs=4)  
    chain_finite[(lat(x) for x in range(L))] = onsite_ZKM
    chain_finite[lat.neighbors()] = hopping_ZKM 
    return chain_finite    

def plot_wave_function_real(syst, params, n):
    """
    Plot the real part of the nth-wavefunction living in the system.
    """
    ham = syst.hamiltonian_submatrix(params=params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    # Sort according to spin up electrons
    eigenvectors = eigenvectors[::4, :]
    plt.plot(np.real(eigenvectors[:, n])) 

def plot_wave_function_imaginary(syst, params, n):
    """
    Plot the real part of the nth-wavefunction living in the system.
    """
    ham = syst.hamiltonian_submatrix(params=params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    # Sort according to spin up electrons
    eigenvectors = eigenvectors[::4, :]
    plt.plot(np.imag(eigenvectors[:, n])) 

def main():
    # without crossing 
    t = 1
    mu = 2*t
    Delta_0 = 0.4*t
    Delta_1 = 0.2*t
    lambda_R = 0.5*t
    
    # with crossing
    # t = 1
    # t_J = t
    # mu = t
    # Delta_0 = 0.4*t
    # Delta_1 = 0.4*t
    # lambda_R = 0.5*t
    
    L = 100
    params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
                  lambda_R=lambda_R, L=L)
    k = -np.pi
    chain = make_chain_finite(**params, k=k)
    chain = chain.finalized()
    #probability
    fig, ax = plt.subplots(dpi=300)
    params["k"] = k
    global eigenvector
    eigenvector = plot_density(chain, params, n=1)
    
    #real part of the wavefunction
    fig, ax = plt.subplots(dpi=300)
    plot_wave_function_real(chain, params, n=0)
    plt.title("Real part of the wavefunction for spin up electron")
    
    #imaginary part of the wavefunction
    fig, ax = plt.subplots(dpi=300)
    plot_wave_function_imaginary(chain, params, n=0)
    plt.title("Imaginary part of the wavefunction for spin up electron")
    
#%%
if __name__ == '__main__':
    main()
