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
    if site.pos < [0.0]:
        theta = 0
    return ( (-2*t*np.cos(k) - mu) * np.kron(tau_z, np.eye(2)) +
            (Delta_0 + 2*Delta_1*np.cos(k)) * np.kron(tau_x, np.eye(2)) +
            (-2)*lambda_R*np.sin(k) * ( np.cos(theta)*np.kron(tau_z, sigma_x) +
            np.sin(theta)*np.kron(tau_z, sigma_y) ) )

def hopping_Josephson_ZKM(site1, site2, t, Delta_0, Delta_1,lambda_R, phi, k, t_J, theta):
    if (site1.pos == [0.0] and site2.pos == [-1.0]) or (site1.pos == [-1.0] and site2.pos == [0.0]):
        return t_J * (np.kron( (tau_z + np.eye(2))/2*np.exp(1j*phi/2) + 
                               (tau_z - np.eye(2))/2*np.exp(-1j*phi/2),
                               (np.eye(2) + sigma_z)/2*np.exp(1j*theta/2) + 
                               (np.eye(2) - sigma_z)/2*np.exp(-1j*theta/2) ) )
    else:
        return ( -t * np.kron(tau_z, np.eye(2)) +
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
    Josephson_junction_ZKM[(lat(x) for x in range(-L, L))] = onsite_Josephson_ZKM
    Josephson_junction_ZKM[lat.neighbors()] = hopping_Josephson_ZKM
    return Josephson_junction_ZKM

def Josephson_current(syst, params):
    fundamental_energy = []
    dphi = np.diff(params["phi"])[0]
    for phi in params["phi"]:
        params["phi"] = phi
        H = syst.hamiltonian_submatrix(params=params)
        eigenvalues, eigenvectors = np.linalg.eig(H)
        fundamental_energy.append(-np.sum(eigenvalues, where=eigenvalues>0) / 2)
    current = np.diff(fundamental_energy) / dphi
    return current

def plot_spectrum(params, k):
    """
    Plot the spectrum by changing the parameter 'phi'.
    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        Finite Kitaev chain.
    phi : np.array
        Phase difference between superconductors.
    """
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Spectrum for H_ZKM")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$J_k$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    ribbon_ZKM = make_Josephson_junction_ZKM()
    ribbon_ZKM = ribbon_ZKM.finalized()
    phi = np.linspace(0, 2*np.pi, 240)
    params["k"] = k
    kwant.plotter.spectrum(ribbon_ZKM, ("phi", phi), params=params, ax=ax)

def plot_k_resolved_current(t, t_J, mu, Delta_0, Delta_1, lambda_R,
                            theta, L, k):
    fig, ax = plt.subplots(dpi=300)
    ax.set_title(rf"k-resolved Josephson current for H_ZKM, $\theta$={theta:.2f}")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$J_k$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    ribbon_ZKM = make_Josephson_junction_ZKM(mu=mu, L=L)
    ribbon_ZKM = ribbon_ZKM.finalized()
    #phi = np.linspace(0, 2*np.pi, 240)
    phi = np.linspace(0, 2*np.pi, 50)
    total_current = []
    for k_value in k:
        params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
                      lambda_R=lambda_R, L=L, phi=phi, k=k_value,
                      t_J=t_J, theta=theta)
        current = Josephson_current(ribbon_ZKM, params)
        total_current.append(current)
        if k_value in [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]:
            ax.plot(phi[:-1], current, label=f"{k_value:.2f}")
        else:
            ax.plot(phi[:-1], current, label="_nolegend_")
    plt.legend(loc="upper right")
    return total_current
    
def plot_total_current(t, t_J, mu, Delta_0, Delta_1, lambda_R,
                            theta, L, k):
    fig, ax = plt.subplots(dpi=300)
    ax.set_title(rf"Total Josephson current for H_ZKM, L={L}")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$J_k$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    ribbon_ZKM = make_Josephson_junction_ZKM(mu=mu, L=L)
    ribbon_ZKM = ribbon_ZKM.finalized()
    phi = np.linspace(0, 2*np.pi, 100)
    for theta_value in theta:
        total_current = []
        for k_value in k:
            params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
                          lambda_R=lambda_R, L=L, phi=phi, k=k_value,
                          t_J=t_J, theta=theta_value)
            current = Josephson_current(ribbon_ZKM, params)
            total_current.append(current)
        total_current = np.sum(total_current, axis=0) / len(k)
        ax.plot(phi[:-1], total_current, label=rf"$\theta={theta_value:.2f}$")
    plt.legend(loc="upper right")
    return fig, ax

def reshape_dat(current, phi, name):
    """
    Saves a .txt file with the right shape for grace
    """
    x, y = np.shape(current)
    phi_list = list(phi[:-1])    #because of np.diff
    phi_list += (y-1)*(["\n"] + phi_list)    #I extend the phi list
    phi = np.array(phi_list)
    phi = np.reshape(phi, (x+(x+1)*(y-1),1))
    current_bis = np.reshape(np.real(current), (x*y,1), order="F")
    current_bis_list = []
    for item in current_bis:
        current_bis_list.append(item[0])
    for i in np.arange(x, x*y, x+1):
        current_bis_list.insert(i, "\n")
    current_bis = np.array(current_bis_list)
    current_bis = np.reshape(current_bis, (x+(x+1)*(y-1),1))
    result = np.append(phi, current_bis, axis=1)
    np.savetxt(name, result, fmt="%s")


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
    
    theta = np.pi
    L = 100
    # k = np.linspace(-np.pi, 0, 240)
    k = np.linspace(-np.pi, -np.pi+0.1, 10)

    params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1,
                  lambda_R=lambda_R, L=L,
                  t_J=t_J, theta=theta)
    
    current = plot_k_resolved_current(k=k, **params)
    #np.save(f"k_current_L=100_no_crossing_theta={theta:.2}", current)
    #fig.savefig(os.getcwd()+f"/Images/Tilted/Josephson_H_ZKM_crossing_L={L}_theta={theta:.2f}.png")

    params.pop("theta")
    theta = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4]
    #fig, ax = plot_total_current(k=k, theta=theta, **params)
    #fig.savefig(os.getcwd()+f"/Images/Tilted/Total_Josephson_H_ZKM_without_crossing_L={L}.png")
    
    
if __name__ == '__main__':
    main()
    print('\007')  # Ending bell

