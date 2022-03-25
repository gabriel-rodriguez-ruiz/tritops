# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 06:53:27 2022

@author: gabri
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

def Hamiltonian(t, k, mu, L, Delta_0, Delta_1, lambda_R):
    r"""Returns the H_k matrix for ZKM model with:

    .. math::
        H_{ZKM} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0+\Delta_k\tau_x\sigma_0
            -2\lambda\sin(k)\tau_z\sigma_z\right]\vec{c}_n+
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0-i\lambda\tau_z\sigma_x + \Delta_1\tau_x\sigma_0 )\vec{c}_{n+1}
            + H.c.
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    chi_k = -mu - 2*t * np.cos(k)
    Delta_k = Delta_0 + 2*Delta_1*np.cos(k)
    onsite = chi_k * np.kron(tau_z, sigma_0) + \
            Delta_k * np.kron(tau_x, sigma_0) - \
            2*lambda_R*np.sin(k) * np.kron(tau_z, sigma_x)
    hopping = -t*np.kron(tau_z, sigma_0) - 1j*lambda_R * np.kron(tau_z, sigma_z) + Delta_1*np.kron(tau_x, sigma_0)
    matrix_diagonal = np.kron(np.eye(L), onsite)     #diagonal part of matrix
    matrix_outside_diagonal = np.block([ [np.zeros((4*(L-1),4)),np.kron(np.eye(L-1), hopping)],
                                         [np.zeros((4,4*L))] ])     #upper diagonal part
    matrix = (matrix_diagonal + matrix_outside_diagonal + matrix_outside_diagonal.conj().T)
    return matrix
# without crossing 
t = 1
mu = -2*t
Delta_0 = -0.4*t
Delta_1 = 0.2*t
lambda_R = 0.5*t
k = 0+0.05*np.pi
L = 200
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L, k=k)

H = Hamiltonian(**params)
eigenvalues, eigenvectors = np.linalg.eigh(H)
# Sort according to the absolute values of energy
eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))]

#%% Spectrum
def spectrum(k_values, **params):
    """Returns an array whose rows are the eigenvalues of the system for
    for a definite k.
    """
    eigenvalues = []
    for k in k_values:
        params["k"] = k
        H = Hamiltonian(**params)
        energies = np.linalg.eigvalsh(H)
        energies = list(energies)
        eigenvalues.append(energies)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues
k = np.linspace(0, np.pi, 100)
spectrum = spectrum(k, **params)  	
#%% Plotting

fig, ax = plt.subplots(figsize=(4,3))
#fig.set_figwidth(246/72)    # in inches, \columnwith=246pt and 1pt=1/72 inch
plt.rc('font', family='serif')  #set font family
plt.rc('xtick', labelsize='small') # reduced tick label size
plt.rc('ytick', labelsize='small')
plt.rc('text', usetex=True)
ax.plot(k, spectrum, marker=".", markersize=0.5,
        color="m") #each column in spectrum is a separate dataset
ax.plot(k, spectrum[:,398:402], marker=".", markersize=0.5,
        color="c") #each column in spectrum is a separate dataset

ax.set_ylim((-3, 3))
ax.set_xlim((0, np.pi))
ax.set_xticks(np.arange(0,1.2,step=0.2)*np.pi)
ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.2,1,step=0.2),1), dtype=str))+["1"])
ax.set_xticks(np.arange(0,1.1,step=0.1)*np.pi)
ax.set_xlabel(r"$k_z/\pi$")
ax.set_ylabel(r"$E(k_z)$")
ax.set_yticks(np.arange(-2,3,step=1))
plt.tight_layout()