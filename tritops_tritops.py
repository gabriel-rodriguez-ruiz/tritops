#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:02:09 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def J(phi, k, v=1):
    t = -1
    return 2*t**2 * np.cos(phi/2) * np.sin(phi/2) / np.sqrt((v*k)**2 + t**2*np.cos(phi/2)**2)

def plot_Josephson_current(v=1):
    """
    Low energy Hamiltonian for tritops-tritops
    junction.
    v = 0 for H0
    v = 1 for H+
    """
    fig, ax = plt.subplots(dpi=300)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$J(\phi)$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    phi = np.linspace(0, 2*np.pi, 1000)
    if v != 0:
        for k in np.linspace(0, 2*np.pi, 100):
            ax.plot(phi, J(phi, k, v))
            ax.set_title("Josephson current for H+")
            fig.savefig(os.getcwd()+"/Images/Josephson_H+_low_energy")
    else:
        ax.plot(phi, J(phi, 0, v))
        ax.set_title("Josephson current for H0")
        fig.savefig(os.getcwd()+"/Images/Josephson_H0_low_energy")

