#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:00:03 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import wave_function, Hamiltonian_A1u, spectrum

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])



#%% Spectrum
t = 1
Delta = 1  #1
mu = -3     #mu = -3  entre -4t y 4t hay estados de borde
k = np.linspace(0, np.pi, 150)
#k = np.linspace(-np.pi, 0, 75)
L = 200

params = dict(t=t, mu=mu, Delta=    Delta,
              L=L)

spectrum_A1u = spectrum(Hamiltonian_A1u, k, **params)

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
    k, spectrum_A1u, linewidth=0.1, color="m"
)  # each column in spectrum is a separate dataset
ax.plot(
    k[:45], spectrum_A1u[:45, 398:402], linewidth=1, color="c"
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
    H_S1 = Hamiltonian_A1u(t, k, mu, L, Delta)
    H_S2 = Hamiltonian_A1u(t, k, mu, L, Delta)
    block_diagonal_matrix = np.block([[H_S1, np.zeros((4*L,4*L))],
                             [np.zeros((4*L,4*L)), H_S2]]) 
    tau_phi = (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    block_diagonal_matrix[4*(L-1):4*L, 4*L:4*(L+1)] = t_J*tau_phi
    block_diagonal_matrix[4*L:4*(L+1), 4*(L-1):4*L] = t_J*tau_phi.conj().T
    return block_diagonal_matrix

def phi_spectrum(Junction, k_value, phi_values, **params):
    """Returns an array whose rows are the eigenvalues of the junction (with function Junction) for
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

def Josephson_current(Junction, k_values, phi_values, **params): 
    """ Returns an array whose columns are the Josephson current for
    a definite phi_value and the rows are the current for
    a definite k_value.
    """    
    dphi = np.diff(phi_values)[0]
    current = []
    for k in k_values:
        fundamental_energy = []
        eigenvalues_phi = phi_spectrum(Junction, k, phi_values, **params) #each row are the energies for a definite phi
        for i in range(len(phi_values)):
            fundamental_energy.append(-np.sum(eigenvalues_phi[i,:], where=eigenvalues_phi[i,:]>0))  #the fundamental energy for each phi
        current.append(list(np.gradient(fundamental_energy, dphi)))
    current = np.array(current)
    return current, fundamental_energy

t = 1
t_J = 0.5
Delta = 1
mu = -3
phi = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)
#k = np.linspace(0, np.pi, 75)
k = [np.pi/4]
#k = np.array([0, 0.01, 0.02])*np.pi
#k = np.linspace(-3, -, 5)

L = 200

params = dict(t=t, mu=mu, Delta=Delta,
              L=L, phi=phi, t_J=t_J)

current, fundamental_energy = Josephson_current(Junction, k, phi, **params)
print('\007')  # Ending bell

#%%
phi = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)

#plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax.plot(phi, current.T, linewidth=0.1)
ax.set_xlabel(r"$\Phi/\pi$")
ax.set_ylabel(r"$J(k)$")
plt.tight_layout()
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
k_value = 0.01
eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_A1u(t=t, k=k_value, mu=mu, L=L, Delta=Delta))
eigenvectors = eigenvectors[:,(2*L-2):(2*L+2)]
#eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))]

plt.figure()
plt.plot(np.abs(eigenvectors[:,0]))

#%% Effective current

def effective_current_A1u(k, phi, t_J, Delta):
    """
    Effective current for A1u.
    """
    E_k_plus = np.sqrt( (Delta*k)**2 + t_J**2*np.cos(phi/2)**2 + 2*t_J*Delta*k*np.cos(phi/2))
    E_k_minus = np.sqrt( (Delta*k)**2 + t_J**2*np.cos(phi/2)**2 - 2*t_J*Delta*k*np.cos(phi/2))
    J_plus = (t_J**2*np.cos(phi/2) + t_J*Delta*k)/E_k_plus
    J_minus = (t_J**2*np.cos(phi/2) - t_J*Delta*k)/E_k_minus
    return -1/2*np.sin(phi/2)*(J_plus+J_minus)

k = np.linspace(0, 0.1*np.pi, 10)
phi = np.linspace(0, 2*np.pi, 240)
t_J = 0.5
Delta = 1
plt.figure()
for k in k: 
    plt.plot(phi, [effective_current_A1u(k, phi, t_J, Delta)for phi in phi])
    
#%% phi_spectrum
plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="xx-large")  # reduced tick label size
plt.rc("ytick", labelsize="xx-large")
plt.rc("axes", labelsize="xx-large")

plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False
plt.figure()
L = 50
k = np.linspace(0, 0.1*np.pi, 10)
phi = np.linspace(0, 2*np.pi, 240)
t_J = 1
Delta = 1
mu = -3    #mu=-3
k_value = 0.1*np.pi

plt.plot(phi, phi_spectrum(Junction, k_value, phi_values=phi, t_J=t_J, t=t, mu=mu, Delta=Delta,
              L=L), color="m", linewidth=0.1)
plt.plot(phi, phi_spectrum(Junction, k_value, phi_values=phi, t_J=t_J, t=t, mu=mu, Delta=Delta,
              L=L)[:,198:202], color="c", linewidth=2)
plt.plot(phi, phi_spectrum(Junction, k_value, phi_values=phi, t_J=t_J, t=t, mu=mu, Delta=Delta,
              L=L)[:,196:198], color="r",linewidth=2)
plt.plot(phi, phi_spectrum(Junction, k_value, phi_values=phi, t_J=t_J, t=t, mu=mu, Delta=Delta,
              L=L)[:,202:204], color="r",linewidth=2)


plt.xlim([0,np.pi])
plt.xlabel(r"$\phi/\pi$")
plt.ylabel(r"$\epsilon_{k=0\pi,n}$")
plt.xticks([0,np.pi/2,np.pi,3/2*np.pi,2*np.pi], ["0","0.5","1","1.5","2"])
plt.tight_layout()

#%% Fundamental energy

fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax.plot(phi/np.pi, fundamental_energy)
ax.set_xlabel(r"$\phi/\pi$")
ax.set_ylabel(r"$E_0(k=0,\phi)$")
plt.tight_layout()

#%% Localization
L = 50
k = np.linspace(0, 0.1*np.pi, 10)
phi = np.linspace(0, 2*np.pi, 240)
t_J = 1
Delta = 1
mu = -3    #mu=-3
k_value = 0*np.pi
phi = 0
eigenvalues, eigenvectors = np.linalg.eigh(Junction(t, k_value, mu, L, Delta, phi, t_J))
zero_modes = eigenvectors[:, 4*L-2:4*L+2]      #4 (2) modes with zero energy (with Zeeman)
andreev_modes_negative = eigenvectors[:, 4*L-4:4*L-2]
andreev_modes_positive = eigenvectors[:, 4*L-4:4*L-2]

plt.plot(np.abs(zero_modes[:,0])**2)
#plt.plot(np.abs(andreev_modes_negative[:,1])**2)
