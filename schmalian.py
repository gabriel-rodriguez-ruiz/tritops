# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:53:47 2022

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
import kwant
import os

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

def hopping_x(site1, site2, t, Delta):
    return ( -t * np.kron(tau_z, np.eye(2)) +
            1j * Delta * np.kron(tau_x, sigma_y) )
# I take the Hamiltonian with +

def hopping_y(site1, site2, t, Delta):
    return ( -t * np.kron(tau_z, np.eye(2))
            + 1j * Delta * np.kron(tau_x, sigma_x) )

def hopping_x_0(site1, site2, t, Delta):
    return ( -t * np.kron(tau_z, np.eye(2)) +
            1j * Delta * np.kron(tau_x, sigma_z) )
# I take the Hamiltonian with +

def hopping_y_0(site1, site2, t, Delta):
    return ( -t * np.kron(tau_z, np.eye(2))
            + 1j * Delta * np.kron(tau_x, sigma_z) )


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

def make_ribbon_0(mu=0, t=1, Delta=1, L=25):
    """
    2D TRITOPS system based on [Schmalian] equation 2
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
    syst[kwant.HoppingKind((1, 0), lat)] = hopping_x_0
    syst[kwant.HoppingKind((0, 1), lat)] = hopping_y_0
    #Fill the target ribbon inside the template syst
    ribbon = kwant.Builder(kwant.TranslationalSymmetry([0,1]))
    ribbon.fill(syst, shape=(lambda site: 0 <= site.pos[0] < L), start=[0, 0])
    return ribbon

def onsite_ZKM(site, mu, Delta_0):
    return -mu * np.kron(sigma_z, np.eye(2)) + Delta_0 * np.kron(tau_x, np.eye(2))

def hopping_ZKM_x(site1, site2, t, lambda_R, Delta_1):     #np.kron() is the tensor product
    return (-t * np.kron(sigma_z, np.eye(2)) - lambda_R * np.kron(sigma_z, 1j*sigma_z)
            + Delta_1 * np.kron(tau_x, np.eye(2)))

def hopping_ZKM_y(site1, site2, t, lambda_R, Delta_1):     #np.kron() is the tensor product
    return (-t * np.kron(sigma_z, np.eye(2)) + lambda_R * np.kron(1j*tau_z, sigma_x)
            + Delta_1 * np.kron(tau_x, np.eye(2)))

def make_ribbon_ZKM(mu=0, t=1, Delta_0=-0.4, Delta_1=0.8, lambda_R=0.5, L=25):
    """
    2D TRITOPS system based on [Schmalian] equation 3
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
    syst[lat(0,0)] = onsite_ZKM
    syst[kwant.HoppingKind((1, 0), lat)] = hopping_ZKM_x
    syst[kwant.HoppingKind((0, 1), lat)] = hopping_ZKM_y
    #Fill the target ribbon inside the template syst
    ribbon = kwant.Builder(kwant.TranslationalSymmetry([0,1]))
    ribbon.fill(syst, shape=(lambda site: 0 <= site.pos[0] < L), start=[0, 0])
    return ribbon

def energy_bands(k_x, t, mu, Lambda):
    return [ -t * ( np.cos(k_x) + 1 ) - mu/2 -
            2*Lambda * np.sin(k_x),
            -t * ( np.cos(k_x) + 1 ) - mu/2 +
            2*Lambda * np.sin(k_x)]

#%% Energy bands
def plot_energy_bands():
    """
    Figure 2a from [Schmalian].
    """
    k_x = np.linspace(-np.pi, np.pi, 1000)
    fig, ax = plt.subplots(dpi=300)
    ax.plot(k_x, [energy_bands(k_x, t=1, mu=-2, Lambda=0.5)[0] for k_x in k_x], label="+")
    ax.plot(k_x, [energy_bands(k_x, t=1, mu=-2, Lambda=0.5)[1] for k_x in k_x], label="-")
    ax.grid()
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$E(k_x)$")
    ax.set_xticks([-np.pi, -np.pi/2,0, np.pi/2, np.pi],
               [r"$-\pi$",r"$-\frac{\pi}{2}$",r"0", r"$\frac{\pi}{2}$", "$\pi$"])
    ax.legend()
    fig.savefig("C:\\Users\\gabri\\OneDrive\\Doctorado\\Python\\Tritops\\Images\\Energy_bands.png")
    
def onsite_Josephson_pm(site, mu, t, k):
    return ( (-2*t*np.cos(k) - mu) * np.kron(tau_z, np.eye(2)) )

def hopping_Josephson_pm(site1, site2, t, Delta, phi):
    if (site1.pos == [0.0] and site2.pos == [-1.0]) or (site1.pos == [-1.0] and site2.pos == [0.0]):
        return 2*t * (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                    + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    else:
        return ( -t * np.kron(tau_z, np.eye(2)) +
                1j * Delta * np.kron(tau_x, sigma_y) )
        

def make_Josephson_junction_pm(t=1, mu=0, Delta=1, L=25, phi=0):
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
    Josephson_junction_pm = kwant.Builder()
    lat = kwant.lattice.chain(norbs=4)  
    # The superconducting order parameter couples electron and hole orbitals
    # on each site, and hence enters as an onsite potential
    # There are L sites in each superconductor
    Josephson_junction_pm[(lat(x) for x in range(-L, L))] = onsite_Josephson_pm
    # Hoppings
    Josephson_junction_pm[lat.neighbors()] = hopping_Josephson_pm
    return Josephson_junction_pm

def Josephson_current(syst, params):
    fundamental_energy = []
    for phi in params["phi"]:
        params["phi"] = phi
        H = syst.hamiltonian_submatrix(params=params)
        eigenvalues, eigenvectors = np.linalg.eig(H)
        fundamental_energy.append(-np.sum(eigenvalues, where=eigenvalues>0) / 2)
    current = np.diff(fundamental_energy)
    return current

#%%
def main():
    #Hamiltonian +-
    mu = 3
    t = -1
    Delta = 0.5
    ribbon_pm = make_ribbon_pm(mu=mu, L=50)
    # Check that the system looks as intended.
    #kwant.plot(ribbon_pm)
    # Finalize the system.
    ribbon_pm = ribbon_pm.finalized()
    # We should see the energy bands.
    params = dict(mu=mu, t=t, Delta=Delta)
    momenta = np.linspace(0, np.pi, 1000)
    fig, ax = plt.subplots(dpi=300)
    kwant.plotter.bands(ribbon_pm, momenta=momenta, params=params, ax=ax)
    ax.set_title(f"Ribbon +- with mu={params['mu']}, t={params['t']}, Delta={params['Delta']}")
    ax.set_xticks([0, np.pi/2, np.pi],
               [r"0", r"$\frac{\pi}{2}$", "$\pi$"])
    ax.grid()
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$E(k_y)$")
    #fig.savefig(os.path.join(path, "Images", f"Ribbon_pm_mu={params['mu']}_t={params['t']}_Delta={params['Delta']}.png"))
    fig.savefig(f"C:\\Users\\gabri\\OneDrive\\Doctorado\\Python\\Tritops\\Images\\Ribbon_pm_mu={params['mu']}_t={params['t']}_Delta={params['Delta']}.png")
    
    #Hamiltonian 0
    mu = 3
    t = -1
    Delta = 0.5
    ribbon_0 = make_ribbon_0(mu=mu, L=50)
    # Check that the system looks as intended.
    #kwant.plot(ribbon_pm)
    # Finalize the system.
    ribbon_0 = ribbon_0.finalized()
    # We should see the energy bands.
    params = dict(mu=mu, t=t, Delta=Delta)
    momenta = np.linspace(0, np.pi, 1000)
    fig, ax = plt.subplots(dpi=300)
    kwant.plotter.bands(ribbon_0, momenta=momenta, params=params, ax=ax)
    ax.set_title(f"Ribbon 0 with mu={params['mu']}, t={params['t']}, Delta={params['Delta']}")
    ax.set_xticks([0, np.pi/2, np.pi],
               [r"0", r"$\frac{\pi}{2}$", "$\pi$"])
    ax.grid()
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$E(k_y)$")
    #fig.savefig(os.path.join(path, "Images", f"Ribbon_0_mu={params['mu']}_t={params['t']}_Delta={params['Delta']}.png"))
    fig.savefig(f"C:\\Users\\gabri\\OneDrive\\Doctorado\\Python\\Tritops\\Images\\Ribbon_0_mu={params['mu']}_t={params['t']}_Delta={params['Delta']}.png")
    fig.savefig("C:\\Users\\gabri\\OneDrive\\Doctorado\\Python\\Tritops\\fig1.pdf")
    
    #Hamiltonian ZKM
    t = 1
    mu = -2*t
    Delta_0 = -0.4*t
    Delta_1 = 0.2*t
    lambda_R = 0.5*t
    L = 50
    ribbon_ZKM = make_ribbon_ZKM(mu=mu, t=t, lambda_R=lambda_R, Delta_0=Delta_0, Delta_1=Delta_1, L=L)
    # Check that the system looks as intended.
    #kwant.plot(ribbon_pm)
    # Finalize the system.
    ribbon_ZKM = ribbon_ZKM.finalized()
    # We should see the energy bands.
    params = dict(mu=mu, t=t, lambda_R=lambda_R, Delta_0=Delta_0, Delta_1=Delta_1)
    momenta = np.linspace(0, np.pi, 1000)
    fig, ax = plt.subplots(dpi=300)
    kwant.plotter.bands(ribbon_ZKM, momenta=momenta, params=params, ax=ax)
    ax.set_title(f"Ribbon ZKM with mu={params['mu']}, lambda_R={params['lambda_R']},t={params['t']}, Delta_0={params['Delta_0']}, Delta_1={params['Delta_1']}")
    ax.set_xticks([0, np.pi/2, np.pi],
               [r"0", r"$\frac{\pi}{2}$", "$\pi$"])
    ax.grid()
    ax.set_xlabel(r"$k_z$")
    ax.set_ylabel(r"$E(k_z)$")
    ax.set_ylim((-4, 4))
    #fig.savefig(os.path.join(path, "Images", f"Ribbon_0_mu={params['mu']}_t={params['t']}_Delta={params['Delta']}.png"))
    fig.savefig(f"C:\\Users\\gabri\\Mi unidad\\Doctorado\\Python\\Tritops\\Images\\Ribbon_ZKM_mu={params['mu']}_lambda_R={params['lambda_R']}_t={params['t']}_Delta_0={params['Delta_0']}_Delta_1={params['Delta_1']}.png")

def main_Josephson():
    #Hamiltonian +-
    mu = 3
    t = -1
    Delta = 0.5
    L = 10
    fig, ax = plt.subplots()
    ax.set_title("k-resolved Josephson current for H+")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$J_k$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
    ax.set_xticklabels([r"0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$","$2\pi$"])
    ax.grid()
    plt.tight_layout()
    syst = make_Josephson_junction_pm(L=L)
    #kwant.plot(syst, site_color=site_color, hop_color=hop_color)
    syst = syst.finalized()
    phi = np.linspace(0, 2*np.pi, 100)
    for k in np.linspace(0, 2*np.pi, 50):
        params = dict(t=t, mu=mu, Delta=Delta, L=L, phi=phi, k=k)
        #plot_spectrum(kitaev, mu)
        current = Josephson_current(syst, params)
        ax.plot(phi[:-1], current)
        #ax.plot(phi[:-1], current, label=f"{k:.2f}")        #plot as function of the phase difference
        #energy = Josephson_current(kitaev, params)
    for k in [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]:
        params = dict(t=t, mu=mu, Delta=Delta, L=L, phi=phi, k=k)
        current = Josephson_current(syst, params)
        ax.plot(phi[:-1], current, label=f"{k:.2f}")        #plot as function of the phase difference
    plt.legend()
    print('\007')  # Ending bell


if __name__ == '__main__':
    #main()
    main_Josephson()