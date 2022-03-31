#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:45:56 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

lambda_R = 0.5
t = 1
mu = (-2*t-2*lambda_R + -2*t+2*lambda_R)/2  #en el medio entre los dos mu_critico
k_x = np.linspace(-np.pi, np.pi)
k_z = np.linspace(-np.pi, np.pi)
X, Y = np.meshgrid(k_x, k_z)
# F is one side of the equation, G is the other
F = 2*t*(np.cos(X) + np.cos(Y)) + mu
G = 2*lambda_R*np.sqrt(np.sin(X)**2 + np.sin(Y)**2)
# Fermi energy
ax.contour(X, Y, F-G, [0], colors="blue")
ax.contour(X, Y, F+G, [0], colors="red")
# nodal surface of pairing potential
ax.contour(X, Y, np.cos(X)+np.cos(Y)-1, [0], colors="black")

ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_z$")
ax.set_yticks(np.linspace(-np.pi, np.pi, 5))
ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
