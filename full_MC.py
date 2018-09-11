import tqdm
import numpy as np
import torch as tch
from gaussian_model import CenteredGM, generate_pd_matrix

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as minimize
from multiprocessing import Pool as ThreadPool
from itertools import repeat


params = {
        # Simulation parameters
        'n_neurons': 50,
        'alpha': 2.,
        'gamma': 100000,
        'beta': None,
        't_max': 5000,
        'n_seeds': 5,
        'n_samples': 10000,


        # Multi-threading params
        'n_threads': 5,
        'silent': False
        }

if params['beta'] is None:
    params['beta'] = 0.0001 * params['n_neurons'] ** (-2)


# def free_energy(N, spectrum, mu, gamma):
#     # Basic free energy
#     return .5 * (mu / np.sqrt(gamma) - tch.log(mu-spectrum).mean().item())


def compute_energy(J, C, params):
    N = params['n_neurons']
    p = params['n_samples']
    gamma = params['gamma']
    beta = params['beta']

    # For energy, eigenvectors are useless
    spectrum, _ = tch.symeig(J, eigenvectors=True)
    # print('new loop')
    # print(spectrum.cpu().numpy())
    # print(type(spectrum))
    # print(len(spectrum))


    # why 0.25 and not 0.5?
    # Understand all relevant scalings
    l2_penalty = 0.25 * (spectrum**2).mean()
    # print(tch.log(spectrum))
    likelihood_energy = 0.5 * (p / N) * (tch.trace(J*C) + tch.log(spectrum).sum())
    # print(tch.trace(J*C), 'trace term')
    # print(tch.log(spectrum).mean(), 'log term')

    # print(l2_penalty * gamma * beta, beta * likelihood_energy)

    return beta * (l2_penalty * gamma + likelihood_energy)


def run_one_thread(C, params, seed):
    # Make experiments different
    np.random.seed(seed)
    # Parameters unpacking
    N = params['n_neurons']
    t_max = params['t_max']
    beta = params['beta']

    # Initialization for J (sym, gaussian)
    J = tch.from_numpy(generate_pd_matrix(N))

    # Initialize the accumulators
    energy_acc = np.zeros(t_max)
    move_acc = np.zeros(t_max)
    thermal_move_acc = np.zeros(t_max)
    eigenvalues_acc = np.zeros((t_max, N))

    energy_acc[0] = compute_energy(J, C, params)

    # MC loop
    for t in tqdm.tqdm(range(1, t_max)):
        # Propose a change
        i, j = np.random.randint(N, size=2)
        epsilon = np.random.normal(scale=1./np.sqrt(N))

        J_prop = J.clone()
        J_prop[i, j] += epsilon
        J_prop[j, i] += epsilon

        F_prop = compute_energy(J_prop, C, params)

        if np.isnan(F_prop):
            print("Invalid move")
            continue

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
    model_to_fit = CenteredGM(params['n_neurons'])
    C_emp = tch.from_numpy(model_to_fit.get_empirical_C(n_samples=params['n_samples']))
    pool = ThreadPool(params['n_threads'])
    results = pool.starmap(run_one_thread, zip(
            [C_emp for _ in range(params['n_seeds'])],
            [params for _ in range(params['n_seeds'])],
            range(params['n_seeds']))
            )
    energies = np.array([tup[0] for tup in results])

    plt.figure()
    plt.errorbar(range(energies.shape[1]), np.mean(energies, axis=0), yerr=np.std(energies, axis=0))
    plt.savefig('out/energy_averaged.pdf')


run_multi_threaded(params)