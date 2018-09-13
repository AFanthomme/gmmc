import numpy as np
import matplotlib.pyplot as plt

def do_analysis():
    energies, eigenvalues = np.load('out/energies.npy'), np.load('out/eigenvalues.npy')

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
    n_bins=100
    mixed_evs = eigenvalues.reshape(eigenvalues.shape[0], -1)
    bounds = (np.min(mixed_evs), np.max(mixed_evs))
    diff_map = np.zeros((eigenvalues.shape[0], n_bins))
    for t in range(eigenvalues.shape[0]):
        for ev in mixed_evs[t]:
            diff_map[t, int((n_bins-1) * (ev-bounds[0]) / (bounds[1] - bounds[0]))] += 1
        diff_map[t] /= np.sum(diff_map[t])

    plt.figure()
    plt.imshow(diff_map.T, aspect='auto')
    plt.savefig('out/eigenvalues_diffmap.pdf')
