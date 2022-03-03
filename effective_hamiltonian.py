#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:12:14 2022

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np

def effective_current(k, phi):
    """
    Equation (43) of [Schmalian] for the effective
    current for the ZKM model.
    """
    
phi = np.linspace(0, 2*np.pi, 1000)
#plt.plot(phi,  np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) )
plt.plot(phi,  -np.sqrt( (np.cos(phi/2) + np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) )
#plt.plot(phi,  np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2 ) )
plt.plot(phi,  -np.sqrt( (np.cos(phi/2) - np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) )

plt.figure()
#The derivative dy/dx is np.diff(y)/np.diff(x)
dphi = np.diff(phi)[0]
plt.plot(phi[:-1], np.diff(-np.sqrt( (np.cos(phi/2) + np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) )/dphi)
plt.plot(phi[:-1], np.diff(-np.sqrt( (np.cos(phi/2) - np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) )/dphi)

plt.plot(phi[:-1], np.diff(-np.sqrt( (np.cos(phi/2) + np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 )  - np.sqrt( (np.cos(phi/2) - np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) )/dphi)

plt.plot(phi, 1/2*np.sin(phi/2)*np.sign(np.cos(phi/2))*( ( 2*abs(np.cos(phi/2)) + 1 )/
                        (np.sqrt( (np.cos(phi/2) + np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) ) +
                        ( 2*abs(np.cos(phi/2)) - 1 )/
                        (np.sqrt( (np.cos(phi/2) - np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) ) ))
