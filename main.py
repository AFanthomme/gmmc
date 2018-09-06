import numpy as np
import torch as tch
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as minimize


params = {'n_spins': 40,
        'alpha': 2.,
        'gamma': 0.3,
        'beta': None,
        't_max': 50000,
        }

if params['beta'] is None:
    params['beta'] = 100 * params['n_spins'] ** 2


def free_energy(N, spectrum, mu, gamma):
    # Basic free energy
    return .5 * (mu / np.sqrt(gamma) - np.mean(np.log(mu-spectrum)))


def compute_energy(J, params):
    N = params['n_spins']
    alpha, gamma = params['alpha'], params['gamma']
    spectrum, _ = np.linalg.eig(J)
    energy = 0.25 * np.mean(spectrum**2)
    mu = minimize(lambda m: free_energy(N, spectrum, m,  gamma), bounds=(np.max(spectrum), 15), method='bounded').x
    return energy + alpha * free_energy(N, spectrum, mu,  gamma)


def run_monte_carlo(params):
    # Parameters unpacking
    N = params['n_spins']
    t_max = params['t_max']
    beta = params['beta']

    # Initialization for J (sym, gaussian and right norm)
    J = np.random.normal(scale=1./np.sqrt(N), size=(N,N))
    J = (J + J.T) / np.sqrt(2)
    J -= np.diag(np.diag(J))

    # Initialize the accumulators
    energy_acc = np.zeros(t_max)
    move_acc = np.zeros(t_max)
    thermal_move_acc = np.zeros(t_max)

    energy_acc[0] = compute_energy(J, params)

    # MC loop
    for t in tqdm.tqdm(range(1, t_max)):
        # Propose a change
        i, j = np.random.randint(N, size=2)
        epsilon = np.random.normal(scale=1./np.sqrt(N))

        J_prop = np.copy(J)
        J_prop[i, j] += epsilon
        J_prop[j, i] += epsilon

        F_prop = compute_energy(J_prop, params)

        delta_F = F_prop - energy_acc[t-1]

        if delta_F < 0:
            energy_acc[t] = F_prop
            J = np.copy(J_prop)
            move_acc[t] = 1
        elif np.random.rand() < np.exp(-beta*delta_F):
            energy_acc[t] = F_prop
            J = np.copy(J_prop)
            move_acc[t] = 1
            thermal_move_acc[t] = 1
        else:
            energy_acc[t] = energy_acc[t-1]

    plt.figure()
    plt.plot(np.cumsum(move_acc), label='Energy moves')
    plt.plot(np.cumsum(thermal_move_acc), label='Entropy moves')
    plt.plot(np.cumsum(thermal_move_acc + move_acc), label='Total moves')
    plt.legend()

    plt.figure()
    plt.plot(energy_acc)

    plt.show()

run_monte_carlo(params)
