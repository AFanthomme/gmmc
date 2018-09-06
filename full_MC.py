import tqdm
import numpy as np
import torch as tch
from gaussian_model import CenteredGM

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as minimize
from multiprocessing import Pool as ThreadPool
from itertools import repeat


params = {
        # Simulation parameters
        'n_spins': 100,
        'alpha': 2.,
        'gamma': 0.3,
        'beta': None,
        't_max': 50000,
        'n_seeds': 15,


        # Multi-threading params
        'n_threads': 12,
        'silent': False
        }

if params['beta'] is None:
    params['beta'] = 15 * params['n_spins'] ** 2


def free_energy(N, spectrum, mu, gamma):
    # Basic free energy
    return .5 * (mu / np.sqrt(gamma) - tch.log(mu-spectrum).mean().item())


def compute_energy(J, params):
    N = params['n_spins']
    alpha, gamma = params['alpha'], params['gamma']
    spectrum, _ = tch.symeig(J)
    energy = 0.25 * (spectrum**2).mean()
    mu = minimize(lambda m: free_energy(N, spectrum, m,  gamma), bounds=(spectrum.max().item(), 15), method='bounded').x
    return energy + alpha * free_energy(N, spectrum, mu,  gamma)


def run_one_thread(params, seed):
    # Make experiments different
    np.random.seed(seed)
    # Parameters unpacking
    N = params['n_spins']
    t_max = params['t_max']
    beta = params['beta']

    # Initialization for J (sym, gaussian and right norm)
    J = tch.normal(tch.zeros([N,N], dtype=tch.float32), 1./np.sqrt(N))
    J = (J + J.t()) / np.sqrt(2)
    J -= tch.diag(tch.diag(J))

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

        J_prop = J.clone()
        J_prop[i, j] += epsilon
        J_prop[j, i] += epsilon

        F_prop = compute_energy(J_prop, params)

        delta_F = F_prop - energy_acc[t-1]

        if delta_F < 0:
            energy_acc[t] = F_prop
            J = J_prop.clone()
            move_acc[t] = 1
        elif np.random.rand() < np.exp(-beta*delta_F):
            energy_acc[t] = F_prop
            J = J_prop.clone()
            move_acc[t] = 1
            thermal_move_acc[t] = 1
        else:
            energy_acc[t] = energy_acc[t-1]

    if not params['silent']:
        plt.figure()
        plt.plot(np.cumsum(move_acc), label='Energy moves')
        plt.plot(np.cumsum(thermal_move_acc), label='Entropy moves')
        plt.plot(np.cumsum(thermal_move_acc + move_acc), label='Total moves')
        plt.legend()
        plt.savefig('out/moves_thread_{}.pdf'.format(seed))

        plt.figure()
        plt.plot(energy_acc)
        plt.savefig('out/energy_thread_{}.pdf'.format(seed))

    return energy_acc, move_acc

def run_multi_threaded(params):
    pool = ThreadPool(params['n_threads'])
    results = pool.starmap(run_one_thread, zip([params for _ in range(params['n_seeds'])], range(params['n_seeds'])))
    energies = np.array([tup[0] for tup in results])

    plt.figure()
    plt.errorbar(range(energies.shape[1]), np.mean(energies, axis=0), yerr=np.std(energies, axis=0))
    plt.savefig('out/energy_averaged.pdf')


run_multi_threaded(params)
