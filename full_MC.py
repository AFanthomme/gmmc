import tqdm
import numpy as np
import torch as tch
from gaussian_model import CenteredGM, generate_pd_matrix

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as minimize
from multiprocessing import Pool as ThreadPool
from itertools import repeat
from analysis import post_run_parsing

import hashlib
import json
import os

def get_id_for_dict(in_dict):
    # forget n_seeds and n_threads, they should not impact the name of exp
    # In particular, raises error if relaunching why changing only n_threads
    dict_filtered = { key: in_dict[key] for key in in_dict.keys() if key not in ['n_threads', 'n_seeds']}
    return hashlib.sha256(json.dumps(dict_filtered, sort_keys=True).encode('utf-8')).hexdigest()[:16]

params = {
        # Simulation parameters
        'n_neurons': 50,
        'alpha': 3.,
        'gamma': 1,
        'beta_normalized': 1.,
        't_max': 400000,
        'n_seeds': 8,

        # method used to select reference C
        'which_C': 'random',

        # Multi-threading and IO params
        'n_threads': 8,
        'silent': False,
        'save_every': 5000,
        }

# Setting secondary parameters values
params['n_samples'] = int(params['alpha'] * params['n_neurons'])
params['beta'] = params['beta_normalized'] * params['n_neurons'] ** 2




def compute_energy(J, C, params):
    N = params['n_neurons']
    p = params['n_samples']
    gamma = params['gamma']
    beta = params['beta']
    alpha = params['alpha']

    # J is the off-diagonal part
    spectrum, _ = tch.symeig(J, eigenvectors=True)

    # THE N² scaling will come from beta, so here everything should be order 1
    # Penalty part is straightforward
    l2_penalty = gamma * 0.25 * (spectrum**2).mean()

    # Likelihood part requires some calculations because of the spherical constraint:
    def surrogate(mu):
        return  mu - tch.log(mu-spectrum).mean().item()

    mu_opt = minimize(surrogate, bounds=(spectrum.max().item(), spectrum.max().item() + 15), method='bounded').x


    likelihood_energy = 0.5 * alpha * (tch.trace(tch.mm(J,C)) / N + surrogate(mu_opt))
    return l2_penalty + likelihood_energy, mu_opt - spectrum


def run_one_thread(out_dir, C, params, seed):
    # Make experiments different
    out_dir += 'seed_{}/'.format(seed)

    # If this fails, the whole exp should fail
    os.makedirs(out_dir)


    np.random.seed(seed)
    # Parameters unpacking
    N = params['n_neurons']
    t_max = params['t_max']
    beta = params['beta']
    save_every = params['save_every']

    # Initialization for J (sym, gaussian, zero diag, sum or eigenvalues equal to N)
    J = tch.normal(mean=tch.zeros([N,N]), std=1./np.sqrt(N) * tch.ones([N,N]))
    J = (J + J.t()) / np.sqrt(2)
    J -= tch.diag(tch.diag(J))

    # Initially, the sum of eigenvalues is very far from N
    # But during MC, it gradually decreases toward correct values
    # print(J)
    # print((tch.symeig(J)[0]))
    # print((tch.symeig(J)[0]).sum())


    # Initialize the accumulators
    energy_acc = np.zeros(t_max)
    move_acc = np.zeros(t_max)
    thermal_move_acc = np.zeros(t_max)
    eigenvalues_acc = np.zeros((t_max, N))

    # The eigenvalues we store are those of J, not (mu*In - J), is it better?
    energy, eigenvalues = compute_energy(J, C, params)
    energy_acc[0] = energy
    eigenvalues_acc[0] = eigenvalues

    # MC loop
    for t in tqdm.tqdm(range(1, t_max)):
        # Propose a change
        i, j = np.random.randint(N, size=2)
        epsilon = np.random.normal(scale=1./np.sqrt(N))

        # To check evolution of sum of ev of J (off-diagonal)
        # print((tch.symeig(J)[0]).sum())

        J_prop = J.clone()
        J_prop[i, j] += epsilon
        J_prop[j, i] += epsilon

        F_prop, spectrum = compute_energy(J_prop, C, params)

        if np.isnan(F_prop):
            print("Invalid move")
            continue

        delta_F = F_prop - energy_acc[t-1]
        # To check that beta delta_f is reasonable
        # print(-beta*delta_F)

        if delta_F < 0:
            energy_acc[t] = F_prop
            J = J_prop.clone()
            move_acc[t] = 1
            eigenvalues_acc[t] = spectrum
        elif np.random.rand() < np.exp(-beta*delta_F):
            energy_acc[t] = F_prop
            J = J_prop.clone()
            move_acc[t] = 1
            thermal_move_acc[t] = 1
            eigenvalues_acc[t] = spectrum
        else:
            energy_acc[t] = energy_acc[t-1]
            eigenvalues_acc[t] = eigenvalues_acc[t-1]

        # Introduce checkpoints !
        if t % save_every == (save_every - 1):
            np.save(out_dir+'energy_acc_ckpt', energy_acc)
            plt.figure()
            plt.plot(energy_acc[:t])
            plt.savefig(out_dir+'energy_real_time.png')
            plt.close()
            np.save(out_dir+'eigenvalues_acc_ckpt', eigenvalues_acc)

    np.save(out_dir+'energy_acc', energy_acc)
    np.save(out_dir+'eigenvalues_acc', eigenvalues_acc)

    return energy_acc, move_acc, eigenvalues_acc

def run_multi_threaded(params):
    # Determine the hash for that particular experiment
    hash = get_id_for_dict(params)
    print('Exp with id {} and params {}'.format(hash, params))
    out_dir = 'out/raw/{}'.format(hash)

    # Just in case two params gave exactly the same hash
    try:
        os.makedirs(out_dir)
        out_dir += '/'
    except FileExistsError:
        out_dir += '_dup'
        os.makedirs(out_dir)
        out_dir += '/'

    with open(out_dir + 'params', 'w') as outfile:
        json.dump(params, outfile)

    if params['which_C'] == 'random':
        model_to_fit = CenteredGM(params['n_neurons'])
    elif params['which_C'] == 'bidiagonal':
        raise NotImplementedError
        # sigma = #0 on the diag, 1 on the two off-diags
        model_to_fit = CenteredGM(params['n_neurons'], sigma=sigma)

    C_emp = tch.from_numpy(model_to_fit.get_empirical_C(n_samples=params['n_samples']).astype(np.float32))

    pool = ThreadPool(params['n_threads'])

    results = pool.starmap(run_one_thread, zip(
            [out_dir for _ in range(params['n_seeds'])],
            [C_emp for _ in range(params['n_seeds'])],
            [params for _ in range(params['n_seeds'])],
            range(params['n_seeds']))
            )



    energies = np.array([tup[0] for tup in results])
    eigenvalues = np.array([tup[2] for tup in results])

    # Switch to a (t_max, N, n_seeds) shape
    eigenvalues = eigenvalues.transpose(1, 2, 0)

    np.save(out_dir + 'energies', energies)
    np.save(out_dir + 'eigenvalues', eigenvalues)

    # from analysis import do_analysis
    # do_analysis()



run_multi_threaded(params)
post_run_parsing()
