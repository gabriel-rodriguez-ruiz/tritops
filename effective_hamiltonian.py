#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:12:14 2022

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np

def effective_current(k, phi, theta, w2, lambda_R, t_J, phi_k, rho_k):
    """
    Equation (43) of [Schmalian] for the effective
    current for the ZKM model.
    """
    t_1 = t_J * np.real(w2*np.exp(-1j*(phi_k+theta/2)))
    t_2 = t_J * np.imag(w2*np.exp(-1j*(phi_k+theta/2)))
    E_k = -2*rho_k*lambda_R*np.sin(k)
    E_k_plus = np.sqrt( (t_1 * np.cos(phi/2) + E_k)**2 + t_2**2*np.cos(phi/2)**2 )
    E_k_minus = np.sqrt( (t_1 * np.cos(phi/2) - E_k)**2 + t_2**2*np.cos(phi/2)**2 )
    J_0 = ( ( ( t_1*np.cos(phi/2) + E_k )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_plus +
            ( ( t_1*np.cos(phi/2) - E_k )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_minus )
    return 1/2 * J_0 * np.sin(phi/2)

# def effective_current(k, phi, theta, w2, lambda_R, t_J, phi_k, rho_k):
#     """
#     Equation (43) of [Schmalian] for the effective
#     current for the ZKM model.
#     """
#     t_1 = t_J * np.real(w2*np.exp(-1j*(phi_k)))
#     t_2 = t_J * np.imag(w2*np.exp(-1j*(phi_k)))
#     E_k = -2*rho_k*lambda_R*np.sin(k)
#     cos = np.cos(phi/2)*np.cos(theta/2)
#     sin = np.sin(phi/2)*np.sin(theta/2)
#     E_k_plus = np.sqrt( (t_1*cos + t_2*sin + E_k)**2 + (cos*t_2 - sin*t_1)**2 )
#     E_k_minus = np.sqrt( (t_1*cos + t_2*sin - E_k)**2 + (cos*t_2 - sin*t_1)**2 )
#     J_0 = ( ( ( t_1*np.cos(phi/2) + E_k )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_plus +
#            ( ( t_1*np.cos(phi/2) - E_k )*t_1 + t_2**2 * np.cos(phi/2) )/E_k_minus )
#     return 1/2 * J_0 * np.sin(phi/2)
# k = -np.pi+0.1
# theta = np.pi/2
#plt.plot(phi, [effective_current(k, phi, theta) for phi in phi])

# phi = np.linspace(0, 2*np.pi, 240)
# theta = 0
# for k in np.linspace(-np.pi, 0, 150):
#     plt.plot(phi, [effective_current(k, phi, theta) for phi in phi], label=f"{k:.2f}", linewidth=0.5)
# #plt.legend(loc="upper right")
# plt.title(rf"Effective current for $\theta = {theta:.2f}$")
# plt.xlabel(r"$\phi$")
# plt.ylabel(r"$J_k$")
# plt.grid()
# k = -np.pi/2    
# for theta in np.linspace(0, np.pi, 10):
#     plt.plot(phi, [effective_current(k, phi, theta) for phi in phi])

#%% Determination of w2

# without crossing 
# t = 1
# t_J = t/2
# Delta_0 = 0.4*t
# Delta_1 = 0.2*t
# mu = t*Delta_0/Delta_1
# lambda_R = 0.5*t

#Aligia
t = 1
t_J = t/2
mu = 2*t
Delta_0 = 4*t
Delta_1 = 2.2*t
lambda_R = 7*t

theta = 0
phi = np.linspace(0, 2*np.pi, 240)

def parameters(k, lambda_R, Delta_0, Delta_1, mu, t):
    s = np.sign(lambda_R*Delta_1)
    Delta_k = Delta_0 + 2*Delta_1*np.cos(k)
    chi_k = -2*t*np.cos(k) - mu
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
    N_k = 1/np.sqrt( 2* (abs(alpha_1)**2/(1-abs(z_1)**2) + 
                           abs(alpha_2)**2/(1-abs(z_2)**2) +
                           alpha_1*np.conjugate(alpha_2)/(1-z_1*np.conjugate(z_2)) +
                           alpha_1*np.conjugate(alpha_2)/(1-z_2*np.conjugate(z_2)) +
                           np.conjugate(alpha_1)*alpha_2/(1-np.conjugate(z_1)*z_2))
                       )
    w_k = np.sqrt(2) * N_k * (alpha_1 + alpha_2)
    Z = 2*N_k**2*( alpha_1**2/(1-z_1**2) +
                  alpha_2**2/(1-z_2**2) +
                  (2*alpha_1*alpha_2)/(1-z_1*z_2) )
    rho_k = abs(Z)
    phi_k = np.angle(Z)
    return w_k**2, rho_k, phi_k, z_1, z_2

fig, ax = plt.subplots(figsize=(4,3))
# fig.title(rf"Effective current for $\theta = {theta:.2f}$")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$J_k$")
ax.grid()
ax.set_xlim([0, 2*np.pi])
#k = np.linspace(-np.pi, -np.pi/2, 10)
k = np.linspace(-np.pi, 0, 75)[:20]
#k = [-np.pi, -np.pi+0.001*np.pi, -np.pi+0.002*np.pi]

for k in k:
    effective = ax.plot(phi, [1.25*effective_current(k, phi, theta, t_J=t_J, lambda_R=lambda_R, w2=parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[0],
                                                     phi_k=parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[2],
                                                     rho_k=parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[1]) for phi in phi],
            label=f"{k:.2f}", linestyle="dashed", linewidth=0.5)

#%%
current = np.load("k_current_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_1_tJ_0.5_theta_0.npy")
#current = np.load("k_current_L_200_Delta0_0.4_Delta1_0.2_lambda_0.5_mu_2_tJ_0.5_theta_pi_over_4.npy")

phi = np.linspace(0, 2*np.pi, 240)

#plt.rc('text', usetex=False)
#fig, ax = plt.subplots(figsize=(4,3), dpi=300)
numerical = ax.plot(phi, current[:20].T, linewidth=0.5, label="Numerical")
#ax.set_xlabel(r"$\Phi/\pi$")
#ax.set_ylabel(r"$J(k)$")
# ax.set_xlim((0, 2*np.pi))
# ax.set_xticks(np.arange(0,2.5,step=0.5)*np.pi)
# ax.set_xticklabels(["0"]+list(np.array(np.round(np.arange(0.5,2,step=0.5),1), dtype=str)) + ["2"])
# ax.set_xticks(np.arange(0,2,step=0.25)*np.pi, minor=True)
# ax.set_yticks(np.arange(-0.08,0.1,step=0.04))
# ax.set_yticks(np.arange(-0.08,0.1,step=0.02), minor=True)
plt.tight_layout()
ax.legend([numerical[0], effective[0]], ["Numerical", "Effective"])

#%% Rho_k and phi_k
plt.close("all")
fig, ax = plt.subplots(figsize=(4,3))
# fig.title(rf"Effective current for $\theta = {theta:.2f}$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\rho_k$")
ax.set_xlim([0, 2*np.pi])
#k = np.linspace(-np.pi, -np.pi/2, 10)
k = np.linspace(0, 2*np.pi, 1000)

ax.plot(k, [parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[1] for k in k])

fig, ax = plt.subplots(figsize=(4,3))
# fig.title(rf"Effective current for $\theta = {theta:.2f}$")
ax.set_xlabel(r"$k$")
ax.set_yticks([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi])
ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$-\pi/4$", "0", r"$\pi/4$", r"$\pi/2$", r"$\pi$"])
ax.set_ylabel(r"$\varphi_k$")
#k = np.linspace(-np.pi, -np.pi/2, 10)
k = np.linspace(0, 2*np.pi, 1000)

ax.plot(k, [parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[2] for k in k])

fig, ax = plt.subplots(figsize=(4,3))
# fig.title(rf"Effective current for $\theta = {theta:.2f}$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$z_i$")
ax.set_xlim([0, 2*np.pi])
#k = np.linspace(-np.pi, -np.pi/2, 10)
k = np.linspace(0, 2*np.pi, 1000)

ax.plot(k, [abs(parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[3]) for k in k], label=r"$z_1$")
ax.plot(k, [abs(parameters(k, lambda_R, Delta_0, Delta_1, mu, t)[4]) for k in k], label=r"$z_2$")
ax.legend()