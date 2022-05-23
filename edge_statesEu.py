#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:00:03 2022

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

def Hamiltonian_Eu(t, k, mu, L, Delta):
    r"""Returns the H_k matrix for Eu model with:

    .. math::
        H_{Eu} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_z \right] +
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0 + i\frac{\Delta}{2}\tau_x\sigma_z)\vec{c}_{n+1}
            + H.c.
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    chi_k = -mu - 2*t * np.cos(k)
    onsite = chi_k * np.kron(tau_z, sigma_0) + \
            Delta *np.sin(k)* np.kron(tau_x, sigma_z)
    hopping = -t*np.kron(tau_z, sigma_0) + 1j*Delta/2 * np.kron(tau_x, sigma_z)
    matrix_diagonal = np.kron(np.eye(L), onsite)     #diagonal part of matrix
    matrix_outside_diagonal = np.block([ [np.zeros((4*(L-1),4)),np.kron(np.eye(L-1), hopping)],
                                         [np.zeros((4,4*L))] ])     #upper diagonal part
    matrix = (matrix_diagonal + matrix_outside_diagonal + matrix_outside_diagonal.conj().T)
    return matrix

def spectrum(system, k_values, **params):
    """Returns an array whose rows are the eigenvalues of the system for
    for a definite k. System should be a function that returns an array.
    """
    eigenvalues = []
    for k in k_values:
        params["k"] = k
        H = system(**params)
        energies = np.linalg.eigvalsh(H)
        energies = list(energies)
        eigenvalues.append(energies)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

#%% Spectrum
t = 1
Delta = 1
mu = -3     #mu = -3  entre -4t y 4t hay estados de borde
k = np.linspace(0, np.pi, 150)

L = 200

params = dict(t=t, mu=mu, Delta=    Delta,
              L=L)

spectrum_Eu = spectrum(Hamiltonian_Eu, k, **params)

#%% Plotting of spectrum
plt.close()

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


fig, ax = plt.subplots(figsize=(4, 3))
# fig.set_figwidth(246/72)    # in inches, \columnwith=246pt and 1pt=1/72 inch
ax.plot(
    k, spectrum_Eu, linewidth=0.1, color="m"
)  # each column in spectrum is a separate dataset
ax.plot(
    k[:35], spectrum_Eu[:35, 398:402], linewidth=1, color="c"
)  # each column in spectrum is a separate dataset

ax.set_ylim((-7, 7))
ax.set_xlim((0, np.pi))
ax.set_xticks(np.arange(0, 1.2, step=0.2) * np.pi)
ax.set_xticklabels(
    ["0"] + list(np.array(np.round(np.arange(0.2, 1, step=0.2), 1), dtype=str)) + ["1"])
ax.set_xticks(np.arange(0, 1.1, step=0.1) * np.pi, minor=True)
ax.set_yticks(np.arange(-6, 7, step=2))
ax.set_yticks(np.arange(-6, 7, step=1), minor=True)
ax.set_xlabel(r"$k_y/\pi$")
ax.set_ylabel(r"$E(k_y)$")

plt.tight_layout()

#%% k-resolved Josephson junction

def Junction(t, k, mu, L, Delta, phi, t_J):
    r"""Returns the array for the Hamiltonian of Josephson junction tilted in an angle theta.
    
    .. math::
        H = H_k^{S1} + H_k^{S2} + H_{J,k}
        
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.
    """
    H_S1 = Hamiltonian_Eu(t, k, mu, L, Delta)
    H_S2 = Hamiltonian_Eu(t, k, mu, L, Delta)
    block_diagonal_matrix = np.block([[H_S1, np.zeros((4*L,4*L))],
                             [np.zeros((4*L,4*L)), H_S2]]) 
    tau_phi = (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    block_diagonal_matrix[4*(L-1):4*L, 4*L:4*(L+1)] = t_J*tau_phi
    block_diagonal_matrix[4*L:4*(L+1), 4*(L-1):4*L] = t_J*tau_phi.conj().T
    return block_diagonal_matrix

def phi_spectrum(k_value, phi_values, **params):
    """Returns an array whose rows are the eigenvalues of the junction for
    a definite phi_value given a fixed k_value.
    """
    eigenvalues = []
    params["k"] = k_value
    for phi in phi_values:
        params["phi"] = phi
        H = Junction(**params)
        energies = np.linalg.eigvalsh(H)
        energies = list(energies)
        eigenvalues.append(energies)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

def Josephson_current(k_values, phi_values, **params): 
    """ Returns an array whose columns are the Josephson current for
    a definite phi_value and the rows are the current for
    a definite k_value.
    """    
    dphi = np.diff(phi_values)[0]
    current = []
    for k in k_values:
        fundamental_energy = []
        eigenvalues_phi = phi_spectrum(k, phi_values, **params) #each row are the energies for a definite phi
        for i in range(len(phi)):
            fundamental_energy.append(-np.sum(eigenvalues_phi[i,:], where=eigenvalues_phi[i,:]>0))  #the fundamental energy for each phi
        current.append(list(np.gradient(fundamental_energy, dphi)))
    current = np.array(current)
    return current

t = 1
t_J = 0.5
Delta = 1
mu = -3
phi = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)
k = np.linspace(0, np.pi, 75)
#k = np.array([0])
#k = np.linspace(-3, -, 5)

L = 200

params = dict(t=t, mu=mu, Delta=Delta,
              L=L, phi=phi, t_J=t_J)

current = Josephson_current(k, phi, **params)
print('\007')  # Ending bell

#%%
phi = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)

plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax.plot(phi, current.T, linewidth=0.1)
ax.set_xlabel(r"$\Phi/\pi$")
ax.set_ylabel(r"$J(k)$")
# ax.set_xlim((0, 2*np.pi))
# ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
# ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
# ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
# ax.set_yticks(np.arange(-0.08,0.1,step=0.04))
# ax.set_yticks(np.arange(-0.08,0.1,step=0.02), minor=True)
plt.tight_layout()

#%% Edge states spin structure
t = 1
Delta = 1
mu = -3     #mu = -3  entre -4t y 4t hay estados de borde
k = np.linspace(0, np.pi, 150)

L = 200

params = dict(t=t, mu=mu, Delta=    Delta,
              L=L)
k_value = 0.1*np.pi
eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_Eu(t=t, k=k_value, mu=mu, L=L, Delta=Delta))
eigenvectors = eigenvectors[:,(2*L-2):(2*L+2)]
#eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))]

plt.figure()
plt.plot(np.abs(eigenvectors[:,0]))