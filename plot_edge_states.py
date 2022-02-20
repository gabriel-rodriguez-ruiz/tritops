# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:59:17 2022

@author: gabri
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

#%% Hamiltonians +- and 0

def onsite(site, mu):
    return -mu * np.kron(tau_z, np.eye(2))

def onsite_chain(site, mu, t, k, Delta):
    return ((-mu - 2*t*np.cos(k)) * np.kron(tau_z, np.eye(2)) +
            Delta*np.sin(k) * np.kron(tau_x, sigma_x) )

def hopping_x(site1, site2, t, Delta):
    return ( -t * np.kron(tau_z, np.eye(2)) +
            1j * Delta * np.kron(tau_x, sigma_y) )
# I take the Hamiltonian with +

def hopping_y(site1, site2, t, Delta):
    return ( -t * np.kron(tau_z, np.eye(2))
            + 1j * Delta * np.kron(tau_x, sigma_x) )

def make_ribbon_pm(mu=0, t=1, Delta=1, L=25):
    """
    2D TRITOPS system based on [Schmalian] equation 1
    with finite boundary conditions in x and periodic in y.
    A factor 1/2 is left out. H=1/2 psi* H_BdG psi
    Parameters
    ----------
    mu : float
        Chemical potential.
    t : float
        Hopping parameter.
    Delta : float
        Pairing potential.
    L : int, optional
        Width of the ribbon. The default is 25.
 
    Returns
    -------
    syst : kwant.builder.Builder
        The representation for the 2D-tritops model.
    """
    #Create a 2D template
    sym = kwant.TranslationalSymmetry((1,0), (0,1))
    syst = kwant.Builder(sym)
    lat = kwant.lattice.square(1, norbs=4)
    syst[lat(0,0)] = onsite
    syst[kwant.HoppingKind((1, 0), lat)] = hopping_x
    syst[kwant.HoppingKind((0, 1), lat)] = hopping_y
    #Fill the target ribbon inside the template syst
    ribbon = kwant.Builder(kwant.TranslationalSymmetry([0,1]))
    ribbon.fill(syst, shape=(lambda site: 0 <= site.pos[0] < L), start=[0, 0])
    return ribbon

def plot_wave_function(syst, params, n):
    """
    Plot the nth-wavefunction living in the system.
    """
    density = kwant.operator.Density(syst)
    ham = syst.hamiltonian_submatrix(params=params)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    eigenvectors = eigenvectors[:, np.argsort(np.abs(eigenvalues))]
    plt.plot(density(eigenvectors[:, n]))   
    kwant.plotter.plot(syst, site_color=density(eigenvectors[:, n]), site_size=0.5,
                       cmap='gist_heat_r')
    
def make_chain_finite(t=1, mu=1, Delta=1, L=25, k=0):
    """
    Create a finite chain.
    
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
    
    Returns
    -------
    kwant.builder.Builder
        The representation for the Kitaev chain.
    
    """
    chain_finite = kwant.Builder()
    lat = kwant.lattice.chain(norbs=4)  
    chain_finite[(lat(x) for x in range(L))] = onsite_chain
    chain_finite[lat.neighbors()] = hopping_x 
    return chain_finite    

def main():
    #Hamiltonian +
    mu = 3
    t = -1
    Delta = 0.5
    L = 100
    ribbon_pm = make_ribbon_pm(mu=mu, L=L)
    # Check that the system looks as intended.
    #kwant.plot(ribbon_pm)
    # Finalize the system.
    ribbon_pm = ribbon_pm.finalized()
    # We should see the energy bands.
    params = dict(mu=mu, t=t, Delta=Delta, k=0)
    momenta = np.linspace(0, np.pi, 1000)
    fig, ax = plt.subplots(dpi=300)
    kwant.plotter.bands(ribbon_pm, momenta=momenta, params=params, ax=ax)
    ax.set_title(f"Ribbon +- with mu={params['mu']}, t={params['t']}, Delta={params['Delta']}")
    ax.set_xticks([0, np.pi/2, np.pi],
               [r"0", r"$\frac{\pi}{2}$", "$\pi$"])
    ax.grid()
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$E(k_y)$")
    
    #Wave function plot in a chain
    k = 0
    chain = make_chain_finite(L=L, mu=mu, t=t, Delta=Delta, k=k)
    chain = chain.finalized()
    fig, ax = plt.subplots(dpi=300)
    params["k"] = k
    plot_wave_function(chain, params, n=0)
    
#%%
if __name__ == '__main__':
    main()

