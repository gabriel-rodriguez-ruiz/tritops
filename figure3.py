# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:02:26 2022

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

current = np.load("Eu_k_current_L_200_mu_-3_Delta_1_t_J=0.5.npy")
phi = np.linspace(0, 2*np.pi, 240)

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


k = np.linspace(0, np.pi, 75)

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(phi, current.T[:,:17], linewidth=0.1, color="m")
ax.plot(phi, current.T[:,17:], linewidth=0.1, color="r")
ax.plot(phi, current.T[:,0], linewidth=1, color="c")

ax.set_xlabel(r"$\phi/\pi$")
ax.set_ylabel(r"$J_k(\phi)[e/\hbar]$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((-0.35, 0.35))
ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax.set_xticklabels(["0"]+["0.5", "1", "1.5"]+ ["2"])
ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax.set_yticks(np.arange(-0.3,0.35,step=0.1))
ax.set_yticks(np.arange(-0.3,0.3,step=0.5), minor=True)
ax.set_yticklabels(["-0.3"] + ["-0.2", "-0.1"] + ["0"] + ["0.1", "0.2", "0.3"])

plt.tight_layout()

#%%
current = np.load("B1u_k_current_L_200_mu_-3_Delta_1_t_J=0.5.npy")
phi = np.linspace(0, 2*np.pi, 240)

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


k = np.linspace(0, np.pi, 75)

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(phi, current.T[:,:17], linewidth=0.1, color="m")
ax.plot(phi, current.T[:,17:], linewidth=0.1, color="r")
ax.plot(phi, current.T[:,0], linewidth=1, color="c")

ax.set_xlabel(r"$\phi/\pi$")
ax.set_ylabel(r"$J_k(\phi)[e/\hbar]$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((-0.35, 0.35))
ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax.set_xticklabels(["0"]+["0.5", "1", "1.5"]+ ["2"])
ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax.set_yticks(np.arange(-0.3,0.35,step=0.1))
ax.set_yticks(np.arange(-0.3,0.3,step=0.5), minor=True)
ax.set_yticklabels(["-0.3"] + ["-0.2", "-0.1"] + ["0"] + ["0.1", "0.2", "0.3"])

plt.tight_layout()

#%%
current_B1u = np.load("B1u_k_current_L_200_mu_-3_Delta_1_t_J=0.5.npy")
current_Eu = np.load("Eu_k_current_L_200_mu_-3_Delta_1_t_J=0.5.npy")

total_current_B1u_s = np.sum(current_B1u, axis=0)
total_current_Eu_s = np.sum(current_Eu, axis=0)

fig, ax = plt.subplots(figsize=(4,3))
phi = np.linspace(0, 2*np.pi, 240)
ax.plot(phi, total_current_B1u_s, label=r"$A_{1u}$", color="r", linewidth=1)
ax.plot(phi, total_current_Eu_s, label=r"$E_u$", color="m", linewidth=1, linestyle="dashed")

#plt.legend()
ax.set_xlabel(r"$\phi/\pi$")
ax.set_ylabel(r"$J(\phi)[e/\hbar]$")
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((-4.5, 4.5))
ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax.set_xticklabels(["0"]+ ["0.5", "1", "1.5"] + ["2"])
ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax.set_yticks(np.arange(-4,5,step=2))
ax.set_yticks(np.arange(-4,5,step=1), minor=True)
#ax.set_yticklabels(["-1.5"] + ["-1"] + ["-0.5"]+ ["0"] + ["0.5"] + ["1"] + ["1.5"])

plt.tight_layout()