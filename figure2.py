#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:37:29 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import spectrum, Hamiltonian

t = 1
Delta_0 = -0.4 * t
Delta_1 = 0.2 * t
mu = -2 * t
lambda_R = 0.5 * t
k = np.linspace(0, np.pi, 150)
L = 200
theta = 0
params = dict(
    t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1, lambda_R=lambda_R, L=L, theta=theta)

spectrum = spectrum(Hamiltonian, k, **params)

#%% Plotting of spectrum

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
# fig.set_figwidth(246/72)    # in inches, \columnwith=246pt and 1pt=1/72 inch
ax.plot(
    k, spectrum, linewidth=0.5, color="m"
)  # each column in spectrum is a separate dataset
ax.plot(
    k, spectrum[:, 398:402], marker=".", markersize=0.5, color="c"
)  # each column in spectrum is a separate dataset

ax.set_ylim((-3, 3))
ax.set_xlim((0, np.pi))
ax.set_xticks(np.arange(0, 1.2, step=0.2) * np.pi)
ax.set_xticklabels(
    ["0"] + list(np.array(np.round(np.arange(0.2, 1, step=0.2), 1), dtype=str)) + ["1"])
ax.set_xticks(np.arange(0, 1.1, step=0.1) * np.pi, minor=True)
ax.set_yticks(np.arange(-2, 3, step=1))
ax.set_yticks(np.arange(-2, 3, step=0.5), minor=True)
ax.set_xlabel(r"$k_z/\pi$")
ax.set_ylabel(r"$E(k_z)$")

plt.tight_layout()

#%% Plotting bands without pairing

k_x = np.linspace(-np.pi, np.pi)
t = 1
mu = 0
lambda_R = 0.5
E_plus = np.array([-2*t*(np.cos(k_x) + 1) - mu - 2*lambda_R*np.sin(k_x) for k_x in k_x])
E_minus = np.array([-2*t*(np.cos(k_x) + 1) - mu + 2*lambda_R*np.sin(k_x) for k_x in k_x])

fig, ax = plt.subplots(figsize=(4,3), dpi=300)
ax.fill_between(k_x, np.ones(len(k_x))*(-2*t-2*lambda_R), np.ones(len(k_x))*(-2*t+2*lambda_R), color="wheat")
ax.plot(k_x, np.ones(len(k_x))*(-2*t-2*lambda_R), color="k", linewidth=0.8)
ax.plot(k_x, np.ones(len(k_x))*(-2*t+2*lambda_R), color="k", linewidth=0.8)
ax.plot(k_x, E_plus, color="b")
ax.plot(k_x, E_minus, color="r")
plt.arrow(0.4*np.pi, -3.9, 0, 0.6, width = 0.02, color="b")
plt.arrow(0.4*np.pi, -1.4, 0, -0.6, width = 0.02, color="r")
plt.text(0.8*np.pi, -2.8, r"$\mu_c$", size="x-large")

ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\epsilon(k_x)-2\lambda\sin k_x$")
ax.set_xlim([-np.pi, np.pi])
ax.set_xticks(np.arange(-1, 1.5, step=0.5) * np.pi)
ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
ax.set_ylim([-5, 1])

#%% All together
grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
fig = plt.figure(figsize=(4,3), dpi=300)
ax1 = fig.add_subplot(grid[0:,0])
ax2 = fig.add_subplot(grid[0,1])
ax3 = fig.add_subplot(grid[1,1])

# Bands without pairing

ax1.fill_between(k_x, np.ones(len(k_x))*(-2*t-2*lambda_R), np.ones(len(k_x))*(-2*t+2*lambda_R), color="wheat")
ax1.plot(k_x, np.ones(len(k_x))*(-2*t-2*lambda_R), color="k", linewidth=0.8)
ax1.plot(k_x, np.ones(len(k_x))*(-2*t+2*lambda_R), color="k", linewidth=0.8)
ax1.plot(k_x, E_plus, color="b")
ax1.plot(k_x, E_minus, color="r")
ax1.arrow(0.4*np.pi, -3.9, 0, 0.6, width = 0.02, color="b")
ax1.arrow(0.4*np.pi, -1.4, 0, -0.6, width = 0.02, color="r")
ax1.text(0.8*np.pi, -2.8, r"$\mu_c$", size="x-large")

ax1.set_xlabel(r"$k_x/\pi$")
ax1.set_ylabel(r"$\epsilon(k_x)-2\lambda\sin k_x$")
ax1.set_xlim([-np.pi, np.pi])
ax1.set_xticks(np.arange(-1, 1.5, step=0.5) * np.pi)
ax1.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
ax1.set_ylim([-5, 1])

# Spectrum
ax3.plot(
    k, spectrum, linewidth=0.5, color="m")  # each column in spectrum is a separate dataset
ax3.plot(
    k, spectrum[:, 398:402], marker=".", markersize=0.5, color="c")  # each column in spectrum is a separate dataset

ax3.set_ylim((-3, 3))
ax3.set_xlim((0, np.pi))
ax3.set_xticks(np.arange(0, 1.2, step=0.2) * np.pi)
ax3.set_xticklabels(
    ["0"] + list(np.array(np.round(np.arange(0.2, 1, step=0.2), 1), dtype=str)) + ["1"])
ax3.set_xticks(np.arange(0, 1.1, step=0.1) * np.pi, minor=True)
ax3.set_yticks(np.arange(-2, 3, step=1))
ax3.set_yticks(np.arange(-2, 3, step=0.5), minor=True)
ax3.set_xlabel(r"$k_z/\pi$")
ax3.set_ylabel(r"$E(k_z)$")