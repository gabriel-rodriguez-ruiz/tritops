#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:02:09 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import kwant

directory = os.getcwd()
path, file = os.path.split(directory)

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

#%%
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

#%% Josephson for ZKM
def Josephson_current(syst, params):
    fundamental_energy = []
    for phi in params["phi"]:
        params["phi"] = phi
        H = syst.hamiltonian_submatrix(params=params)
        eigenvalues, eigenvectors = np.linalg.eig(H)
        fundamental_energy.append(-np.sum(eigenvalues, where=eigenvalues>0) / 2)
    current = np.diff(fundamental_energy)
    return current

def plot_spectrum(syst, phi, params, ax=None):
    """
    Plot the spectrum by changing the parameter 'phi'.
    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        Finite Kitaev chain.
    phi : np.array
        Phase difference between superconductors.
    """
    kwant.plotter.spectrum(syst, ("phi", phi), params=params, ax=ax)

def onsite_Josephson_ZKM(site, mu, Delta_0, Delta_1, lambda_R, k, t):
    return ( (-2*t*np.cos(k) - mu) * np.kron(tau_z, np.eye(2)) + (Delta_0 + 2*Delta_1*np.cos(k)) * np.kron(tau_x, np.eye(2))  +
            2*lambda_R*np.sin(k) * np.kron(tau_z, sigma_x) )

def hopping_Josephson_ZKM(site1, site2, t, Delta_0, Delta_1,lambda_R, phi, k, t_J):
    if (site1.pos == [0.0] and site2.pos == [-1.0]) or (site1.pos == [-1.0] and site2.pos == [0.0]):
        return t_J * (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                    + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    else:
        return ( -t * np.kron(tau_z, np.eye(2)) -
                1j*lambda_R * np.kron(tau_z, sigma_z) +
                Delta_1 * np.kron(tau_x, np.eye(2)))
                                               

def make_Josephson_junction_ZKM(t=1, mu=0, Delta=1, L=25, phi=0, t_J=1):
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
    Josephson_junction_ZKM[(lat(x) for x in range(-L, L))] = onsite_Josephson_ZKM
    # Hoppings
    Josephson_junction_ZKM[lat.neighbors()] = hopping_Josephson_ZKM
    return Josephson_junction_ZKM

def main():
    # without crossing 
    t = 1
    t_J = t/2
    mu = 2*t
    Delta_0 = 0.4*t
    Delta_1 = 0.2*t
    lambda_R = 0.5*t
    
    # with crossing
    # t = 1
    # t_J = t
    # mu = t
    # Delta_0 = 0.4*t
    # Delta_1 = 0.4*t
    # lambda_R = 0.5*t
    
    L = 10
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("k-resolved Josephson current for H_ZKM")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$J_k$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    ribbon_ZKM = make_Josephson_junction_ZKM(mu=mu, L=L)
    #kwant.plot(syst, site_color=site_color, hop_color=hop_color)
    ribbon_ZKM = ribbon_ZKM.finalized()
    phi = np.linspace(0, 2*np.pi, 100)
    for k in np.linspace(-np.pi, np.pi, 10):
        params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1, lambda_R=lambda_R, L=L, phi=phi, k=k, t_J=t_J)
        #plot_spectrum(kitaev, mu)
        current = Josephson_current(ribbon_ZKM, params)
        ax.plot(phi[:-1], current)
        #ax.plot(phi[:-1], current, label=f"{k:.2f}")
    #plt.legend(loc="upper right")
    #fig.savefig(os.getcwd()+f"/Images/Josephson_H_ZKM_crossing_L={L}")
    
    k = np.pi
    fig, ax = plt.subplots(dpi=300)
    ax.set_title(f"Spectrum for H0 and k={k}")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$E(\phi)$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    ribbon_ZKM = make_Josephson_junction_ZKM(mu=mu, L=L)
    ribbon_ZKM = ribbon_ZKM.finalized()
    params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1, lambda_R=lambda_R, L=L, phi=phi, k=0, t_J=t_J)
    phi = np.linspace(0, 2*np.pi, 50)
    params["phi"] = phi
    plot_spectrum(ribbon_ZKM, phi, params, ax=ax)  

if __name__ == '__main__':
    main()