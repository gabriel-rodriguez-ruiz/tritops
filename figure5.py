#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:32:06 2022

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

L = 200
theta = 0

params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L,
              t_J=t_J, theta=theta, phi=phi)

#current = Josephson_current(k, phi, **params)
print('\007')  # Ending bell

#%%
current = np.load("k_current_L_200_Delta0_0.2_Delta1_0.2_lambda_0.5_mu_1_tJ_1_theta_0.npy")
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

fig = plt.figure(figsize=(4,3), dpi=300)
#fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax1 = fig.add_subplot()
ax1.plot(phi, current.T[:,0], linewidth=1, color="c")
ax1.plot(phi, current.T[:,1:50], linewidth=0.1, color="m")
ax1.plot(phi, current.T[:,50:], linewidth=0.1, color="r")

ax1.set_xlabel(r"$\Phi/\pi$")
ax1.set_ylabel(r"$J(k_z)$")
ax1.set_xlim((0, 2*np.pi))
ax1.set_ylim((-0.15, 0.15))
ax1.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax1.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
ax1.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax1.set_yticks(np.arange(-0.1,0.15,step=0.1))
ax1.set_yticks(np.arange(-0.15,0.2,step=0.05), minor=True)
#plt.tight_layout()

#%% all together

current = np.load("k_current_L_200_Delta0_0.2_Delta1_0.2_lambda_0.5_mu_1_tJ_1_theta_0.npy")
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

fig = plt.figure(figsize=(4,3), dpi=300)
#fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax1 = fig.add_subplot()
ax1.plot(phi, current.T[:,0], linewidth=1, color="c")
ax1.plot(phi, current.T[:,1:50], linewidth=0.1, color="m")
ax1.plot(phi, current.T[:,50:], linewidth=0.1, color="r")

ax1.set_xlabel(r"$\Phi/\pi$", labelpad=0)
ax1.set_ylabel(r"$J(k_z)$", labelpad=0)
ax1.set_xlim((0, 2*np.pi))
ax1.set_ylim((-0.15, 0.15))
ax1.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax1.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
ax1.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax1.set_yticks(np.arange(-0.1,0.15,step=0.1))
ax1.set_yticks(np.arange(-0.15,0.2,step=0.05), minor=True)
ax1.set_yticklabels(["-0.1", "0", "0.1"])
ax1.tick_params(axis='y', which='major', pad=0)
ax1.tick_params(axis='x', which='major', pad=0)
#plt.tight_layout()

current = np.load("k_current_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_1_tJ_0.5_theta_0.npy")
phi = np.linspace(0, 2*np.pi, 240)

ax2 = fig.add_axes([0.62, 0.64, 0.4*9/10, 0.3*9/10])
ax2.plot(phi, current.T[:,0], linewidth=1, color="c")
ax2.plot(phi, current.T[:,1:50], linewidth=0.1, color="m")
ax2.plot(phi, current.T[:,50:], linewidth=0.1, color="r")

ax2.set_xlabel(r"$\Phi/\pi$", labelpad=0)
ax2.set_ylabel(r"$J(k_z)$", labelpad=0)
ax2.set_xlim([0, np.pi])
ax2.set_ylim([0, 0.12])
# ax2.set_xticks(np.arange(0,1.2,step=0.2)*np.pi)
# ax2.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
# ax2.set_xticks(np.arange(0,1,step=0.1)*np.pi, minor=True)
ax2.set_xticks(np.arange(0,1.2,step=0.5)*np.pi)
ax2.set_xticklabels(["0", "0.5", "1"])
ax2.set_xticks(np.arange(0,1,step=0.25)*np.pi, minor=True)


ax2.set_yticks(np.arange(0,0.15,step=0.05))
ax2.set_yticks(np.arange(0,0.15,step=0.025), minor=True)
ax2.set_yticklabels(["0", "0.05", "0.1"])
ax2.tick_params(axis='y', which='major', pad=0)
ax2.tick_params(axis='x', which='major', pad=0)