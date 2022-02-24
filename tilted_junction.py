#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:53:01 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import kwant
import os
import scipy.linalg

directory = os.getcwd()
path, file = os.path.split(directory)

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def onsite_Josephson_ZKM(site, mu, Delta_0, Delta_1, lambda_R, k, t, theta):
    return ( (-2*t*np.cos(k) - mu) * np.kron(tau_z, np.eye(2)) + (Delta_0 + 2*Delta_1*np.cos(k)) * np.kron(tau_x, np.eye(2))  +
            2*lambda_R*np.sin(k) * (np.cos(theta)*np.kron(tau_z, sigma_x) + np.sin(theta)*np.kron(tau_z, sigma_y)))

def hopping_Josephson_ZKM(site1, site2, t, Delta_0, Delta_1,lambda_R, phi, k, t_J):
    if (site1.pos == [0.0] and site2.pos == [-1.0]) or (site1.pos == [-1.0] and site2.pos == [0.0]):
        return t_J * (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                    + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    else:
        return ( -t * np.kron(tau_z, np.eye(2)) -
                1j*lambda_R * np.kron(tau_z, sigma_z) +
                Delta_1 * np.kron(tau_x, np.eye(2)))
    
def make_Josephson_junction_ZKM(t=1, mu=0, Delta=1, L=25, phi=0, t_J=1, theta=0):
    """
    Create a 1D tight-binding model for
    Josephson's junction with magnetic flux.
    
    Parameters
    ----------
    t : float, optional
        Hopping parameter. The default is 1.
    mu : float optional
        Chemical potential. The default is 1.
    Delta : float, optional
        Superconducting gap. The default is 1.
    L : int, optional
        Number of sites in the chain. The default is 25.
    phi : float, optional
        Magnetic flux in units of quantum flux h/(2e).
    
    Returns
    -------
    kwant.builder.Builder
        The representation for the tight-binding model.
    """
    Josephson_junction_ZKM = kwant.Builder()
    lat = kwant.lattice.chain(norbs=4)  
    # The superconducting order parameter couples electron and hole orbitals
    # on each site, and hence enters as an onsite potential
    # There are L sites in each superconductor
    Josephson_junction_ZKM[(lat(x) for x in range(0, L))] = onsite_Josephson_ZKM(theta=0)
    # Hoppings
    Josephson_junction_ZKM[lat.neighbors()] = hopping_Josephson_ZKM
    return Josephson_junction_ZKM
