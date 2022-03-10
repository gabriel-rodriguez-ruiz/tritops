#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:12:14 2022

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np

def effective_current(k, phi, theta, lambda_R=0.5, t_J=0.5, w2=1, phi_k=0, rho_k=1):
    """
    Equation (43) of [Schmalian] for the effective
    current for the ZKM model.
    """
    #w2 = (0.19+1j) / (np.sqrt(0.19**2 + 1**2))
    theta_k = np.arctan(np.real(w2)/np.imag(w2))
    t_1 = 2 * t_J * np.abs(w2) * np.cos(theta_k - phi_k + theta/2)
    t_2 = 2 * t_J * np.abs(w2) * np.sin(theta_k - phi_k + theta/2)
    t_lambda = -2*lambda_R * rho_k *np.sin(k)
    E_k_plus = np.sqrt( (t_1 * np.cos(phi/2) + t_lambda)**2 + t_2**2*np.cos(phi/2)**2 )
    E_k_minus = np.sqrt( (t_1 * np.cos(phi/2) - t_lambda)**2 + t_2**2*np.cos(phi/2)**2 )
    J_0 = ( ( ( t_1*np.cos(phi/2) + t_lambda )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_plus +
           ( ( t_1*np.cos(phi/2) - t_lambda )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_minus )
    return 1/2 * J_0 * np.sin(phi/2)

# k = -np.pi+0.1
# theta = np.pi/2
#plt.plot(phi, [effective_current(k, phi, theta) for phi in phi])

phi = np.linspace(0, 2*np.pi, 240)
theta = 0
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

#%% Determination of w2

# without crossing 
# t = 1
# t_J = t/2
# mu = 2*t
# Delta_0 = 0.4*t
# Delta_1 = 0.2*t
# lambda_R = 0.5*t

#Aligia
t = 1
t_J = t/2
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

def parameters(k):
    s = 1
    Delta_k = Delta_0 + 2*Delta_1*np.cos(k)
    chi_k = -2*t*np.cos(k)
    # Solve a*z**2 + b*z + c = 0
    a = s*Delta_1 + lambda_R - 1j*t
    b = s*Delta_k + 1j*chi_k
    c = s*Delta_1 - lambda_R - 1j*t
    z_1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    z_2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    a_1 = chi_k - (t+1j*lambda_R) * z_1
    a_2 = chi_k - (t+1j*lambda_R) * z_2
    b_1 = Delta_k + Delta_1*z_1
    b_2 = Delta_k + Delta_1*z_2
    alpha_1 = a_2 + 1j*s*b_2
    alpha_2 = -( a_1 + 1j*s*b_1 )
    N_k = np.sqrt( 1/ (2* (abs(alpha_1)**2/(1-abs(z_1)**2) + 
                           abs(alpha_2)**2/(1-abs(z_2)**2) +
                           alpha_1*np.conjugate(alpha_2)/(1-z_1*np.conjugate(z_2)) +
                           alpha_1*np.conjugate(alpha_2)/(1-z_2*np.conjugate(z_2)) +
                           np.conjugate(alpha_1)*alpha_2/(1-np.conjugate(z_1)*z_2))
                       ) )
    w_k = np.sqrt(2) * N_k * (alpha_1 + alpha_2)
    
    Z = 2*N_k**2*( alpha_1**2/(1-z_1**2) +
                  alpha_2**2/(1-z_2**2) +
                  (2*alpha_1*alpha_2)/(1-z_1*z_2) )
    rho_k = abs(Z)
    phi_k = np.angle(Z)
    return w_k**2, rho_k, phi_k


plt.figure()
plt.title(rf"Effective current for $\theta = {theta:.2f}$")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$J_k$")
plt.grid()
for k in np.linspace(-np.pi, -np.pi/2, 50):
    plt.plot(phi, [effective_current(k, phi, theta, w2=parameters(k)[0], phi_k=parameters(k)[2], rho_k=parameters(k)[1]) for phi in phi], label=f"{k:.2f}")
