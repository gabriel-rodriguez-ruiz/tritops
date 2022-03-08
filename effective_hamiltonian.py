#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:12:14 2022

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np

phi = np.linspace(0, 2*np.pi, 1000)
# plt.plot(phi,  -np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) )
# plt.plot(phi,  -np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2 ) )
# plt.plot(phi,  -np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2) -
#                           np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) )
plt.plot(phi,  -np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) )
plt.plot(phi,  -np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2 ) )

plt.plot(phi,  np.sign(np.cos(phi/2)) * ( np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2) -
                        np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) ) )
plt.plot(phi,  -np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2) -
                        np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) )


plt.figure()
#The derivative dy/dx is np.diff(y)/np.diff(x)
dphi = np.diff(phi)[0]
plt.plot(phi[:-1], np.diff(-np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) )/dphi)
plt.plot(phi[:-1], np.diff(-np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2 ) )/dphi)
# Numerical derivative
plt.plot(phi[:-1], np.diff(-np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 )  - np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2 ) )/dphi)
# Analytical derivative
plt.plot(phi, 1/2*np.sin(phi/2)*np.sign(np.cos(phi/2))*( ( 2*abs(np.cos(phi/2)) + 1 )/
                        (np.sqrt( (np.cos(phi/2) + np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) ) +
                        ( 2*abs(np.cos(phi/2)) - 1 )/
                        (np.sqrt( (np.cos(phi/2) - np.sign(np.cos(phi/2)))**2 + np.cos(phi/2)**2 ) ) ))
# plt.plot(phi, 1/2*np.sin(phi/2)*np.sign(np.cos(phi/2))*( ( 2*np.cos(phi/2) + 1 )/
#                         (np.sqrt( (np.cos(phi/2) + 1)**2 + np.cos(phi/2)**2 ) ) +
#                         ( 2*np.cos(phi/2) - 1 )/
#                         (np.sqrt( (np.cos(phi/2) - 1)**2 + np.cos(phi/2)**2 ) ) ))

#%%
def effective_current(k, phi, theta, lambda_R=0.5, t_J=0.5, w2=1):
    """
    Equation (43) of [Schmalian] for the effective
    current for the ZKM model.
    """
    w2 = (0.19+1j) / (np.sqrt(0.19**2 + 1**2))
    theta_k = np.arctan(np.real(w2)/np.imag(w2))
    t_1 = 2 * t_J * np.abs(w2) * np.cos(theta_k+theta/2)
    t_2 = 2 * t_J * np.abs(w2) * np.sin(theta_k+theta/2)
    t_lambda = -2*lambda_R*np.sin(k)
    E_k_plus = np.sqrt( (t_1 * np.cos(phi/2) + t_lambda)**2 + t_2**2*np.cos(phi/2)**2 )
    E_k_minus = np.sqrt( (t_1 * np.cos(phi/2) - t_lambda)**2 + t_2**2*np.cos(phi/2)**2 )
    J_0 = ( ( ( t_1*np.cos(phi/2) + t_lambda )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_plus +
           ( ( t_1*np.cos(phi/2) - t_lambda )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_minus )
    return 1/2 * J_0 * np.sin(phi/2)


plt.figure()
phi = np.linspace(0, 2*np.pi, 240)
# k = -np.pi+0.1
# theta = np.pi/2
#plt.plot(phi, [effective_current(k, phi, theta) for phi in phi])


theta = np.pi/2
for k in np.linspace(-np.pi, 0, 150):
    plt.plot(phi, [effective_current(k, phi, theta) for phi in phi], label=f"{k:.2f}", linewidth=0.5)
#plt.legend(loc="upper right")
plt.title(rf"Effective current for $\theta = {theta:.2f}$")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$J_k$")
plt.grid()
# k = -np.pi/2    
# for theta in np.linspace(0, np.pi, 10):
#     plt.plot(phi, [effective_current(k, phi, theta) for phi in phi])

