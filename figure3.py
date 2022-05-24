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