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
#t_J = 5
Delta_0 = 0.4    #0.4
Delta_1 = 0.2     #0.4
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t
#phi = np.linspace(0, 2*np.pi, 240)
phi = np.linspace(0, 2*np.pi, 480)
#phi = np.linspace(0, 0.1, 10)

k = np.linspace(-np.pi, 0, 75)
#k = [-np.pi]

L = 200
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

current = current_A1u_s

phi = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 480)
#phi = np.linspace(0, 0.1, 10)
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
current_A1u_s = np.load("k_current_TRITOPS_S_A1u_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")
current_Eu_s = np.load("k_current_TRITOPS_S_Eu_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")
current_ZKM_s = np.load("k_current_TRITOPS_S_ZKM_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")

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

plt.close()

fig = plt.figure(figsize=(4,6))
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
axs[0].plot(phi, current_A1u_s.T[:, 0], linewidth=1, color="c")
axs[0].plot(phi, current_A1u_s.T[:, 1:30], linewidth=0.1, color="m")
axs[0].plot(phi, current_A1u_s.T[:, 30:], linewidth=0.1, color="r")

axs[1].plot(phi, current_Eu_s.T[:, 30:], linewidth=0.1, color="r")
axs[1].plot(phi, current_Eu_s.T[:, 1:30], linewidth=0.1, color="m")
axs[1].plot(phi, current_Eu_s.T[:, 0], linewidth=1, color="c")

axs[2].plot(phi, current_ZKM_s.T[:, 30:], linewidth=0.1, color="r")
axs[2].plot(phi, current_ZKM_s.T[:, 1:30], linewidth=0.1, color="m")
axs[2].plot(phi, current_ZKM_s.T[:, 0], linewidth=1, color="c")

axs[0].set_xlabel(r"$\phi/\pi$")
axs[0].set_ylabel(r"$J_k(\phi)[e/\hbar]$")
axs[0].set_xlim((0, 2*np.pi))
axs[0].set_ylim((-0.12, 0.12))
axs[0].set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
axs[0].set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
axs[0].set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
axs[0].set_yticks(np.arange(-0.1,0.15,step=0.1))
axs[0].set_yticks(np.arange(-0.1,0.1,step=0.05), minor=True)
axs[0].set_yticklabels(["-0.1"] + ["0"] + ["0.1"])


axs[1].set_xlabel(r"$\phi/\pi$")
axs[1].set_ylabel(r"$J_k(\phi)[e/\hbar]$")
axs[1].set_xlim((0, 2*np.pi))
axs[1].set_ylim((-0.12, 0.12))
axs[1].set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
axs[1].set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
axs[1].set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
axs[1].set_yticks(np.arange(-0.1,0.15,step=0.1))
axs[1].set_yticks(np.arange(-0.1,0.15,step=0.05), minor=True)
axs[1].set_yticklabels(["-0.1"] + ["0"] + ["0.1"])

axs[2].set_xlabel(r"$\phi/\pi$")
axs[2].set_ylabel(r"$J_k(\phi)[e/\hbar]$")
axs[2].set_xlim((0, 2*np.pi))
axs[2].set_ylim((-0.06, 0.06))
axs[2].set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
axs[2].set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
axs[2].set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
axs[2].set_yticks(np.arange(-0.05, 0.06,step=0.05))
axs[2].set_yticks(np.arange(-0.05,0.05,step=0.025), minor=True)
axs[2].set_yticklabels(["-0.05"] + ["0"] + ["0.05"])

plt.tight_layout()

#%% Figure 8, total current

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc('text', usetex=True)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


current_A1u_s = np.load("k_current_TRITOPS_S_A1u_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")
current_Eu_s = np.load("k_current_TRITOPS_S_Eu_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")
current_ZKM_s = np.load("k_current_TRITOPS_S_ZKM_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_0.npy")

total_current_A1u_s = np.sum(current_A1u_s, axis=0)
total_current_Eu_s = np.sum(current_Eu_s, axis=0)
total_current_ZKM_s = np.sum(current_ZKM_s, axis=0)

fig, ax = plt.subplots(figsize=(4,3))
phi = np.linspace(0, 2*np.pi, 240)
ax.plot(phi, total_current_A1u_s, label=r"$A_{1u}$", color="c", linewidth=1)
ax.plot(phi, total_current_Eu_s, label=r"$E_u$", color="m", linewidth=1)
ax.plot(phi, total_current_ZKM_s, label=r"$ZKM$", color="r", linewidth=1)

#plt.legend()
ax.set_xlabel(r"$\phi/\pi$")
ax.set_ylabel(r"$J(\phi)[e/\hbar]$")
ax.set_xlim((0, 2*np.pi))
ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
ax.set_yticks(np.arange(-1.5,2,step=0.5))
ax.set_yticks(np.arange(-1.5,1.5,step=0.25), minor=True)
ax.set_yticklabels(["-1.5"] + ["-1"] + ["-0.5"]+ ["0"] + ["0.5"] + ["1"] + ["1.5"])

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#I want to select the x-range for the zoomed region. I have figured it out suitable values
# by trial and error. How can I pass more elegantly the dates as something like
x1 = 0
x2 = 0.25

# select y-range for zoomed region
y1 = -0.3
y2 = 0.1

# Make the zoom-in plot:
axins = zoomed_inset_axes(ax, 2.5, loc=4) # zoom = 2
axins.plot(phi, total_current_A1u_s, label=r"$A_{1u}$", color="c", linewidth=1)
axins.plot(phi, total_current_Eu_s, label=r"$E_u$", color="m", linewidth=1)
axins.plot(phi, total_current_ZKM_s, label=r"$ZKM$", color="r", linewidth=1)
#Remove ticks
plt.tick_params(right = False)
plt.tick_params(top = False)
plt.tick_params(bottom = False)


axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks([0, -0.25], ["0", ""], visible=True)

plt.tick_params(labelright = False)

box, c1, c2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.indicate_inset_zoom(axins, edgecolor="black")

# Change connector lines to dotted
for c in [c1,c2]:
    c.set_linestyle(":") 
plt.tight_layout()