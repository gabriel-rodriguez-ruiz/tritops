#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:34:48 2022

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

def Hamiltonian(t, k, mu, L, Delta_0, Delta_1, lambda_R, theta):
    r"""Returns the H_k matrix for ZKM model with:

    .. math::
        H_{ZKM} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_{n=1}^L 
            \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0+\Delta_k\tau_x\sigma_0
            -2\lambda\sin(k)\tau_z(cos(\theta)\sigma_x + sin(\theta)\sigma_y)\right]\vec{c}_n+
            \sum_{n=1}^{L-1}             
            \left[
            \vec{c}^\dagger_n(-t\tau_z\sigma_0-i\lambda\tau_z\sigma_z + \Delta_1\tau_x\sigma_0 )\vec{c}_{n+1}
            + H.c.
            \right]
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
       
       \xi_k = -\mu - 2t\cos(k)
       
       \Delta_k = \Delta_0+2\Delta_1\cos(k)
    """
    chi_k = -mu - 2*t * np.cos(k)
    Delta_k = Delta_0 + 2*Delta_1*np.cos(k)
    onsite = chi_k * np.kron(tau_z, sigma_0) + \
            Delta_k * np.kron(tau_x, sigma_0) - \
            2*lambda_R*np.sin(k) * (np.cos(theta)*np.kron(tau_z, sigma_x) + np.sin(theta)*np.kron(tau_z, sigma_y))
    hopping = -t*np.kron(tau_z, sigma_0) - 1j*lambda_R * np.kron(tau_z, sigma_z) + Delta_1*np.kron(tau_x, sigma_0)
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

def wave_function(system, k_values, **params):
    """Returns an array whose first dimension is a 2-D square array with the eigenvectors of the system in columns for
    for a definite k. System should be a function that returns an array.
    """
    eigenvectors = []
    for k in k_values:
        params["k"] = k
        H = system(**params)
        energies, eigenvector = np.linalg.eigh(H)
        eigenvector = list(eigenvector)
        eigenvectors.append(eigenvector)
    eigenvectors = np.array(eigenvectors)
    return energies, eigenvectors

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
    return current

def Hamiltonian_A1u(t, k, mu, L, Delta):
    r"""Returns the H_k matrix for A1u model with:

    .. math::
        H_{A1u} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] +
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c.
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    chi_k = -mu - 2*t * np.cos(k)
    onsite = chi_k * np.kron(tau_z, sigma_0) + \
            Delta *np.sin(k)* np.kron(tau_x, sigma_y)
    hopping = -t*np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x)
    matrix_diagonal = np.kron(np.eye(L), onsite)     #diagonal part of matrix
    matrix_outside_diagonal = np.block([ [np.zeros((4*(L-1),4)),np.kron(np.eye(L-1), hopping)],
                                         [np.zeros((4,4*L))] ])     #upper diagonal part
    matrix = (matrix_diagonal + matrix_outside_diagonal + matrix_outside_diagonal.conj().T)
    return matrix

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

def Junction_ZKM_s(t, k, mu, L, Delta_0, Delta_1, lambda_R, theta, t_J, phi):
    r"""Returns the array for the Hamiltonian of Josephson junction
     (TRITOPS-S) tilted in an angle theta.
    
    .. math::
        H = H_k^{S1} + H_k^{S2} + H_{J,k}
        
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.
    """
    H_S1 = Hamiltonian(t, k, mu, L, Delta_0, Delta_1, lambda_R, theta=0)
    H_S2 = Hamiltonian(t, k, mu, L, Delta_0, 0, 0, theta=theta)
    block_diagonal_matrix = np.block([[H_S1, np.zeros((4*L,4*L))],
                             [np.zeros((4*L,4*L)), H_S2]]) 
    tau_phi = (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    block_diagonal_matrix[4*(L-1):4*L, 4*L:4*(L+1)] = t_J*tau_phi
    block_diagonal_matrix[4*L:4*(L+1), 4*(L-1):4*L] = t_J*tau_phi.conj().T
    return block_diagonal_matrix

def Junction_A1u_s(t, k, mu, L, Delta_0, Delta_1, lambda_R, theta, t_J, phi):
    r"""Returns the array for the Hamiltonian of Josephson junction
     (TRITOPS-S) tilted in an angle theta.
    
    .. math::
        H = H_k^{S1} + H_k^{S2} + H_{J,k}
        
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.
    """
    H_S1 = Hamiltonian_A1u(t, k, mu, L, Delta_0)
    H_S2 = Hamiltonian(t, k, mu, L, Delta_0, 0, 0, theta=theta)
    block_diagonal_matrix = np.block([[H_S1, np.zeros((4*L,4*L))],
                             [np.zeros((4*L,4*L)), H_S2]]) 
    tau_phi = (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    block_diagonal_matrix[4*(L-1):4*L, 4*L:4*(L+1)] = t_J*tau_phi
    block_diagonal_matrix[4*L:4*(L+1), 4*(L-1):4*L] = t_J*tau_phi.conj().T
    return block_diagonal_matrix

def Junction_Eu_s(t, k, mu, L, Delta_0, Delta_1, lambda_R, theta, t_J, phi):
    r"""Returns the array for the Hamiltonian of Josephson junction
     (TRITOPS-S) tilted in an angle theta.
    
    .. math::
        H = H_k^{S1} + H_k^{S2} + H_{J,k}
        
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.
    """
    H_S1 = Hamiltonian_Eu(t, k, mu, L, Delta_0)
    H_S2 = Hamiltonian(t, k, mu, L, Delta_0, 0, 0, theta=theta)
    block_diagonal_matrix = np.block([[H_S1, np.zeros((4*L,4*L))],
                             [np.zeros((4*L,4*L)), H_S2]]) 
    tau_phi = (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    block_diagonal_matrix[4*(L-1):4*L, 4*L:4*(L+1)] = t_J*tau_phi
    block_diagonal_matrix[4*L:4*(L+1), 4*(L-1):4*L] = t_J*tau_phi.conj().T
    return block_diagonal_matrix