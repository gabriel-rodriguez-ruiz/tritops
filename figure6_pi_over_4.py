#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:16:47 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import *

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

#with crossing
t = 1
t_J = t
Delta_0 = 0.2    #0.4
Delta_1 = 0.2     #0.4
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t
phi = np.linspace(0, 2*np.pi, 240)
k = np.linspace(-np.pi, 0, 75)

L = 100
theta = np.pi/4

params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L,
              t_J=t_J, theta=theta, phi=phi)

#current = Josephson_current(k, phi, **params)
print('\007')  # Ending bell

#%%
current = np.load("k_current_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_pi_over_4.npy")
#current_bis = np.load("k_current_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_1_tJ_0.5_theta_0.npy")

phi = np.linspace(0, 2*np.pi, 240)

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc('text', usetex=True)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot()
ax.plot(phi, current.T[:,0], linewidth=1, color="c")
ax.plot(phi, current.T[:,1:20], linewidth=0.1, color="m")
ax.plot(phi, current.T[:,20:], linewidth=0.1, color="r")
# ax.plot(phi, current.T[:,1:50], linewidth=0.1, color="m")
# ax.plot(phi, current.T[:,50:], linewidth=0.1, color="r")

ax.set_xlabel(r"$\Phi/\pi$")
ax.set_ylabel(r"$J(k)$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((-0.15, 0.15))
ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax.set_yticks(np.arange(-0.1,0.15,step=0.1))
ax.set_yticks(np.arange(-0.15,0.2,step=0.05), minor=True)

# ax.set_xlabel(r"$\Phi/\pi$")
# ax.set_ylabel(r"$J(k)$")
# ax.set_xlim((0, 1*np.pi))
# ax.set_ylim((0, 0.15))
# ax.set_xticks(np.arange(0,1.5,step=0.5)*np.pi)
# ax.set_xticklabels(["0"]+ ["0.5"]+ [""])
# ax.set_xticks(np.arange(0,1,step=0.25)*np.pi, minor=True)
# ax.set_yticks(np.arange(-0.1,0.15,step=0.1))
# ax.set_yticks(np.arange(-0.15,0.2,step=0.05), minor=True)

plt.tight_layout()

ax3 = fig.add_subplot()
from effective_hamiltonian import parameters, effective_current

# without crossing 
t = 1
t_J = t/2
Delta_0 = 0.4*t
Delta_1 = 0.2*t
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t

ax3 = fig.add_axes([0.65, 0.65, 0.4*7.5/10, 0.3*8/10])
theta = np.pi/4
phi_values = np.linspace(0, 2*np.pi, 240)
#k = np.linspace(-np.pi, 0, 75)[1:10]
k = np.linspace(-np.pi, 0, 75)[1:11]

for k in k:
    effective = ax3.plot(phi, [1.25*effective_current(k, phi, theta, t_J=t_J, lambda_R=lambda_R, w2=parameters(k)[0], phi_k=parameters(k)[2], rho_k=parameters(k)[1]) for phi in phi_values],
            label=f"{k:.2f}", linestyle = "dashed", linewidth=0.5, color="m")
ax3.plot(phi, [1.25*effective_current(-np.pi, phi, theta, t_J=t_J, lambda_R=lambda_R, w2=parameters(-np.pi)[0], phi_k=parameters(-np.pi)[2], rho_k=parameters(-np.pi)[1]) for phi in phi_values],
        label=f"{k:.2f}", linestyle = "dashed", linewidth=0.5, color="c")

ax3.plot(phi_values, current.T[:,0], linewidth=1, color="c")
ax3.plot(phi_values, current.T[:,1:11], linewidth=0.1, color="m")


ax3.set_xlim(0, np.pi+0.1)
ax3.set_ylim(-0.02, 0.15)
ax3.set_xticks(np.arange(0,1.5,step=0.5)*np.pi)
ax3.set_xticklabels(["0"]+ ["0.5"] + ["1"])
ax3.set_xticks(np.arange(0,1,step=0.25)*np.pi, minor=True)
ax3.set_xlabel(r"$\Phi/\pi$", labelpad=0)
ax3.set_ylabel(r"$J(k)$", labelpad=0)
ax3.set_yticks(np.arange(0,0.15,step=0.1))
ax3.set_yticks(np.arange(0,0.2,step=0.05), minor=True)
ax3.set_yticklabels(["0", "0.1"])
ax3.tick_params(axis='y', which='major', pad=0)
ax3.tick_params(axis='x', which='major', pad=0)
