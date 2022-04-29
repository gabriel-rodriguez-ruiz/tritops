#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:47:13 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import Josephson_current, Junction_ZKM_s, Junction_A1u_s, Junction_Eu_s

# ZKM-S
#with crossing
t = 1
t_J = t/2
Delta_0 = 0.4    #0.4
Delta_1 = 0.2     #0.4
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t
phi = np.linspace(0, 2*np.pi, 240)

k = np.linspace(-np.pi, 0, 75)

L = 100
theta = 0
params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L,
              t_J=t_J, theta=theta, phi=phi)

#%% ZKM-S
current_ZKM_s = Josephson_current(Junction_ZKM_s, k, phi, **params)

print('\007')  # Ending bell

#%% A1u-S

current_A1u_s = Josephson_current(Junction_A1u_s, k, phi, **params)

print('\007')  # Ending bell

#%% Eu-S

current_Eu_s = Josephson_current(Junction_Eu_s, k, phi, **params)

print('\007')  # Ending bell

#%% Plotting

current = current_Eu_s

phi = np.linspace(0, 2*np.pi, 240)
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

#%% all three TRITOPS-S
current_A1u_s = np.load("k_current_TRITOPS_S_A1u_L_100_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")
current_Eu_s = np.load("k_current_TRITOPS_S_Eu_L_100_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")

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

fig = plt.figure(figsize=(4,6), dpi=300)
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
axs[0].plot(phi, current_A1u_s.T, linewidth=0.1)
axs[1].plot(phi, current_Eu_s.T, linewidth=0.1)
#axs[2].plot(x, y, '+')

axs[0].set_xlabel(r"$\Phi/\pi$")
axs[0].set_ylabel(r"$J(k)$")
axs[0].set_xlim((0, 2*np.pi))
axs[0].set_ylim((-0.15, 0.15))
axs[0].set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
axs[0].set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
axs[0].set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
axs[0].set_yticks(np.arange(-0.1,0.15,step=0.1))
axs[0].set_yticks(np.arange(-0.15,0.2,step=0.05), minor=True)

axs[1].set_xlabel(r"$\Phi/\pi$")
axs[1].set_ylabel(r"$J(k)$")
axs[1].set_xlim((0, 2*np.pi))
axs[1].set_ylim((-0.15, 0.15))
axs[1].set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
axs[1].set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
axs[1].set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
axs[1].set_yticks(np.arange(-0.1,0.15,step=0.1))
axs[1].set_yticks(np.arange(-0.15,0.2,step=0.05), minor=True)

