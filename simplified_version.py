#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:45:56 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt


plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc('text', usetex=True)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


fig, ax = plt.subplots(figsize=(4,4), dpi=300)

lambda_R = 0.5
t = 1
mu = (-2*t-2*lambda_R + -2*t+2*lambda_R)/2  #en el medio entre los dos mu_critico
#mu = -2*t+2*lambda_R-0.3
k_x = np.linspace(-np.pi, np.pi)
k_z = np.linspace(-np.pi, np.pi)
X, Y = np.meshgrid(k_x, k_z)
# F is one side of the equation, G is the other
F = -2*t*(np.cos(X) + np.cos(Y)) - mu
G = 2*lambda_R*np.sqrt(np.sin(X)**2 + np.sin(Y)**2)
# Fermi energy
fermi1 = ax.contour(X, Y, F-G, [0], colors="green") #lower band
fermi2 = ax.contour(X, Y, F+G, [0], colors="magenta")   #upper band
# nodal surface of pairing potential
nodal = ax.contour(X, Y, np.cos(X)+np.cos(Y)-1, [0], colors="black")
ax.contourf(X, Y, np.cos(X)+np.cos(Y)-1, [0,1], colors="mistyrose")

ax.contourf(X, Y, np.cos(X)+np.cos(Y)-1, [-10,0], colors="lightskyblue")

ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_z$")
ax.set_yticks(np.linspace(-np.pi, np.pi, 5))
ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

ax.axhline(0, linestyle="--")
nodal_legend,_ = nodal.legend_elements()
fermi1_legend,_ = fermi1.legend_elements()
fermi2_legend,_ = fermi2.legend_elements()

ax.legend([nodal_legend[0], fermi1_legend[0], fermi2_legend[0]],
          [r"$\Delta_{\mathbf{k}}=0$", r"$\epsilon_{F-}$", r"$\epsilon_{F+}$"],
          loc="upper left", ncol=3)
ax.text(-np.pi/2, -3/4*np.pi, r"$-2t-2\lambda<\mu<-2t+2\lambda$")

#Fermi points
ax.contour(X, Y, -mu-2*t*(np.cos(X)+1)+2*lambda_R*np.sin(X), [0], colors="blue")
ax.contour(X, Y, -mu-2*t*(np.cos(X)+1)-2*lambda_R*np.sin(X), [0], colors="red")

plt.tight_layout()