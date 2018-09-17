import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def update_database():
    # Explore the raw folder and build a pandas frame with all relevant info
    hashes = get_immediate_subdirectories('out/raw')
    all_keys = set()
    dict_of_dicts = {}

    for hash in hashes:
        with open('out/raw/{}/params'.format(hash), 'r') as outfile:
            params = json.load(outfile)
        dict_of_dicts[hash] = params

    database = pd.DataFrame.from_dict(dict_of_dicts)
    print(database)
    database.to_pickle('out/raw/parameters_database.pkl')



def do_analysis():
    energies, eigenvalues = np.load('out/energies.npy'), np.load('out/eigenvalues.npy')
    print("Data loaded")

    plt.figure()
    plt.errorbar(range(energies.shape[1]), np.mean(energies, axis=0), yerr=np.std(energies, axis=0))
    plt.savefig('out/energy_averaged.pdf')



    # Sum over evs, (t_max, n_seeds)
    mean_of_evs = eigenvalues.mean(axis=1)
    plt.figure()
    plt.errorbar(range(eigenvalues.shape[0]), np.mean(mean_of_evs, axis=1), yerr=np.std(mean_of_evs, axis=1))
    plt.savefig('out/mean_eigenvalue_averaged.pdf')

    # Make a "diffusion map" for the eigenvalues
    # First, mix the eigenvalues from all seeds :
    n_bins=40
    mixed_evs = eigenvalues.reshape(eigenvalues.shape[0], -1)
    bounds = (np.min(mixed_evs), np.max(mixed_evs))
    diff_map = np.zeros((eigenvalues.shape[0], n_bins))
    for t in range(eigenvalues.shape[0]):
        for ev in mixed_evs[t]:
            diff_map[t, int((n_bins-1) * (ev-bounds[0]) / (bounds[1] - bounds[0]))] += 1
        diff_map[t] /= np.sum(diff_map[t])

    # Also group by close times to get smoother results?

    plt.figure()
    plt.imshow(diff_map.T, extent=[0, diff_map.shape[0], bounds[0], bounds[1]], aspect='auto')
    plt.savefig('out/eigenvalues_diffmap.pdf')

if __name__ == '__main__':
    update_database()
    # do_analysis()
