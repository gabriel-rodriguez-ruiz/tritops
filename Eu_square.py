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

def Hamiltonian_Eu(t, mu, L_x, L_y, Delta):
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
    hopping_x = -t*np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_z)
    hopping_y = -t*np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_z)
    matrix_diagonal = np.kron(np.eye(L_x*L_y), onsite)     #diagonal part of matrix
    matrix_outside_diagonal_x = np.block([ [np.zeros((4*(L_x-1)*L_y,4*L_y)),np.kron(np.eye((L_x-1)*L_y), hopping_x)],
                                         [np.zeros((4*L_y, 4*L_x*L_y))] ])     #upper diagonal part
    Z = np.zeros((L_x*L_y-1, (L_x*L_y-1)))
    np.fill_diagonal(Z, np.append(np.ones(L_y-1), 0))            
    matrix_outside_diagonal_y = np.block([ [np.zeros((4*(L_x*L_y-1),4)),np.kron(Z, hopping_y)],
                                         [np.zeros((4, 4*L_x*L_y))] ])    #upper diagonal part
            #upper diagonal part
    #L_>=3
    matrix = (matrix_diagonal + matrix_outside_diagonal_x + matrix_outside_diagonal_x.conj().T +
              matrix_outside_diagonal_y + matrix_outside_diagonal_y.conj().T)
    return matrix

t = 1
Delta = 1
mu = -2   #mu = -3  entre -4t y 0 hay estados de borde

L_x = 20
L_y = 20

params = dict(t=t, mu=mu, Delta=Delta,
              L_x=L_x, L_y=L_y)

H = Hamiltonian_Eu(t, mu, L_x, L_y, Delta)
eigenvalues, eigenvectors = np.linalg.eigh(H)
zero_modes = eigenvectors[:,(2*L_x*L_y-2):(2*L_x*L_y+2)]    # I extract the eigenvectors asociated to zero energy
zero_mode_up = zero_modes[::4,0].reshape((L_x,L_y)) # I choose one zero mode and reshape to sites in real space
majorana_up_minus = (zero_modes[0::4,0]+1j*zero_modes[3::4,0]).reshape((L_x,L_y))
majorana_up_plus = (zero_modes[0::4,2]+1j*zero_modes[3::4,2]).reshape((L_x,L_y))

majorana_down_minus = (zero_modes[1::4,0]+1j*zero_modes[2::4,0]).reshape((L_x,L_y))
majorana_down_plus = (zero_modes[1::4,2]+1j*zero_modes[2::4,2]).reshape((L_x,L_y))

#plt.imshow(np.abs(zero_mode_up))
#plt.imshow(np.abs(majorana_up))
plt.imshow(np.abs(majorana_up_plus))
plt.colorbar()
plt.title(rf"$\mu$={mu}t")

# plt.figure()
# plt.imshow(np.abs(zero_modes[0::4,2].reshape((L_x,L_y))))
# plt.title("up destruction")
# plt.colorbar()

# plt.figure()
# plt.imshow(np.abs(zero_modes[3::4,2].reshape((L_x,L_y))))
# plt.title("up creation")
# plt.colorbar()

# plt.figure()
# plt.imshow(np.abs(zero_modes[1::4,2].reshape((L_x,L_y))))
# plt.title("down destruction")
# plt.colorbar()


# plt.figure()
# plt.imshow(np.abs(zero_modes[2::4,2].reshape((L_x,L_y))))
# plt.title("down creation")
# plt.colorbar()
    
#plt.figure()
#plt.plot(eigenvalues)
#%%
def plot_majorana(mu, index_zero):
    H = Hamiltonian_Eu(t, mu, L_x, L_y, Delta)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    zero_modes = eigenvectors[:,(2*L_x*L_y-2+index_zero):(2*L_x*L_y+2+index_zero)]    # I extract the eigenvectors asociated to zero energy
    majorana_up_plus = (zero_modes[0::4,2]+1j*zero_modes[3::4,2]).reshape((L_x,L_y))
    plt.imshow(np.abs(majorana_up_plus))
    plt.title(rf"$\mu$={mu}t")

H = Hamiltonian_Eu(t, mu, L_x, L_y, Delta)
eigenvalues, eigenvectors = np.linalg.eigh(H)
fig, axs = plt.subplots(2, 3)
index_zero = [0, 5, 10, 15, 20, 25]
for i in range(len(index_zero)):
    zero_modes = eigenvectors[:,(2*L_x*L_y-2+index_zero[i]):(2*L_x*L_y+2+index_zero[i])]    # I extract the eigenvectors asociated to zero energy
    majorana_up_plus = (zero_modes[0::4,2]+1j*zero_modes[3::4,2]).reshape((L_x,L_y))
    axs[i//3,i%3].imshow(np.abs(majorana_up_plus))
    axs[i//3,i%3].title.set_text(f"E={eigenvalues[2*L_x*L_y+index_zero[i]]:.1e}")