#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:58:20 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import Josephson_current, Junction_ZKM_s, Junction_A1u_s, Junction_Eu_s

# ZKM-S
#with crossing
t = 1
t_J = 5
Delta_0 = 0.4    #0.4
Delta_1 = 0.2     #0.4
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t
phi = np.linspace(0, 0.1*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 480)
#phi = np.linspace(0, 0.1, 10)

#k = np.linspace(-np.pi, 0, 75)
k = [-np.pi]

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

#%% Iteration over t_J A1u

t_J = np.linspace(0.5, 100, 5)
current_A1u_s = []
for t_J_value in t_J:
    params["t_J"] = t_J_value
    current_A1u_s.append(Josephson_current(Junction_A1u_s, k, phi, **params))

print('\007')  # Ending bell

#%%
#plotting A1u

phi = np.linspace(0, 0.1*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 480)
#phi = np.linspace(0, 0.1, 10)
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(4,3), dpi=300)
for i in range(len(t_J)):
    ax.plot(phi, current_A1u_s[i].T, linewidth=0.1, label=f"$t_J$={t_J[i]}")
ax.set_xlabel(r"$\Phi/\pi$")
ax.set_ylabel(r"$J(k)$")
# ax.set_xlim((0, 2*np.pi))
# ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
# ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
# ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
# ax.set_yticks(np.arange(-0.08,0.1,step=0.04))
# ax.set_yticks(np.arange(-0.08,0.1,step=0.02), minor=True)
ax.legend(loc="upper right")
ax.set_title("A1u-S k=0")
plt.tight_layout()

#%% Iteration over t_J ZKM

t_J = np.linspace(0.5, 100, 5)
current_ZKM_s = []
for t_J_value in t_J:
    params["t_J"] = t_J_value
    current_ZKM_s.append(Josephson_current(Junction_ZKM_s, k, phi, **params))

print('\007')  # Ending bell

#%%
#plotting ZKM

phi = np.linspace(0, 0.1*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 480)
#phi = np.linspace(0, 0.1, 10)
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(4,3), dpi=300)
for i in range(len(t_J)):
    ax.plot(phi, current_ZKM_s[i].T, linewidth=0.1, label=f"$t_J$={t_J[i]}")
ax.set_xlabel(r"$\Phi/\pi$")
ax.set_ylabel(r"$J(k)$")
# ax.set_xlim((0, 2*np.pi))
# ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
# ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
# ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
# ax.set_yticks(np.arange(-0.08,0.1,step=0.04))
# ax.set_yticks(np.arange(-0.08,0.1,step=0.02), minor=True)
ax.legend(loc="upper right")
ax.set_title("ZKM-S k=0")
plt.tight_layout()

#%% Iteration over t_J Eu

t_J = np.linspace(0.5, 10, 5)
current_Eu_s = []
for t_J_value in t_J:
    params["t_J"] = t_J_value
    current_Eu_s.append(Josephson_current(Junction_Eu_s, k, phi, **params))

print('\007')  # Ending bell

#%%
#plotting Eu

phi = np.linspace(0, 0.1*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 480)
#phi = np.linspace(0, 0.1, 10)
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(4,3), dpi=300)
for i in range(len(t_J)):
    ax.plot(phi, current_Eu_s[i].T, linewidth=0.1, label=f"$t_J$={t_J[i]}")
ax.set_xlabel(r"$\Phi/\pi$")
ax.set_ylabel(r"$J(k)$")
# ax.set_xlim((0, 2*np.pi))
# ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
# ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
# ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
# ax.set_yticks(np.arange(-0.08,0.1,step=0.04))
# ax.set_yticks(np.arange(-0.08,0.1,step=0.02), minor=True)
ax.legend(loc="upper right")
ax.set_title("Eu-S k=0")
plt.tight_layout()