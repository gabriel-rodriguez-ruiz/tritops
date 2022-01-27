# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:53:47 2022

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
import kwant

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def onsite(site, mu):
    return -mu * np.kron(tau_z, np.eye(2))

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
    Delta_0 : float
        On-site paring potential.
    Delta_1 : float
        Nearest neighbor pairing potential.
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

def main():
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
    ax.set_title(f"Ribbon with mu={params['mu']}, t={params['t']}, Delta={params['Delta']}")
    ax.set_xticks([0, np.pi/2, np.pi],
               [r"0", r"$\frac{\pi}{2}$", "$\pi$"])
    ax.grid()
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$E(k_y)$")
    fig.savefig(f"../Images/Ribbon_mu={params['mu']}_t={params['t']}_Delta={params['Delta']}.png")

if __name__ == '__main__':
    main()
