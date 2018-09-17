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
    database.to_pickle('out/raw/parameters_database.pkl')




def parse_individual_subfolder(subfolder):
    # Automated part of the analysis
    # More challenging analysis should be done somewhere else

    # If we keep adding more points, need to reduce before doing the figure
    energies, eigenvalues = np.load(subfolder + '/energies.npy'), np.load(subfolder + '/eigenvalues.npy')
    print("Data loaded for subfolder {}".format(subfolder))

    plt.figure()
    plt.errorbar(range(energies.shape[1]), np.mean(energies, axis=0), yerr=np.std(energies, axis=0))
    plt.savefig(subfolder + '/energy_averaged.png')



    # Sum over evs, (t_max, n_seeds)
    mean_of_evs = eigenvalues.mean(axis=1)
    plt.figure()
    plt.errorbar(range(eigenvalues.shape[0]), np.mean(mean_of_evs, axis=1), yerr=np.std(mean_of_evs, axis=1))
    plt.savefig(subfolder + '/mean_eigenvalue_averaged.png')

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
    plt.savefig(subfolder + '/eigenvalues_diffmap.png')

    plt.close()

def post_run_parsing():
    update_database()
    exp_dirs = get_immediate_subdirectories('out/raw')
    for exp_dir in exp_dirs:
        parse_individual_subfolder('out/raw/' + exp_dir)

if __name__ == '__main__':
    post_run_parsing()
    # do_analysis()
