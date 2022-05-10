# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 06:53:27 2022

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import Hamiltonian, spectrum, Junction, phi_spectrum, Josephson_current

#with crossing
# t = 1
# t_J = t/2
# Delta_0 = 0.4    #0.4
# Delta_1 = 0.2     #0.4
# mu = t*Delta_0/Delta_1
# lambda_R = 0.5*t

#Aligia
# t = 1
# t_J = t/2
# mu = 2*t
# Delta_0 = 4*t
# Delta_1 = 2.2*t
# lambda_R = 7*t

# without crossing 
t = 1
t_J = t/2
Delta_0 = 0.4*t
Delta_1 = 0.2*t
mu = t*Delta_0/Delta_1
lambda_R = 0.5*t    #lambda_R=0.5*t

phi = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)
k = np.linspace(-np.pi, 0, 75)
#k = np.linspace(-np.pi, 0, 10)

L = 200
theta = np.pi/4

params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
              lambda_R=lambda_R, L=L,
              t_J=t_J, theta=theta, phi=phi)

current = Josephson_current(Junction, k, phi, **params)
print('\007')  # Ending bell

#%%
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
