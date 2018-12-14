import numpy as np
# import torch as tch
# from torch.distributions.multivariate_normal import MultivariateNormal


def generate_random_matrix(dim):
    sigma = np.random.normal(np.zeros([dim,dim], dtype=np.float32), 1./np.sqrt(dim))
    sigma = (sigma + sigma.T) / np.sqrt(2)
    return sigma


def normalize(precision_matrix):
    # Impose PD and diagonal of inverse equal 1 -> covariance of all x equal to 1
    # The diagonal of precision matrix is forced to 0
    dim = precision_matrix.shape[0]

    precision_matrix -= np.diag(np.diag(precision_matrix))

    offset = np.max(np.linalg.eig(precision_matrix)[0]) + 0.01
    precision_matrix = np.diag([offset for _ in range(dim)]) - precision_matrix
    mean_of_eigs = np.mean(np.diag(np.linalg.inv(precision_matrix)))
    precision_matrix *= mean_of_eigs

    return np.real(precision_matrix)


class CenteredGM:
    def __init__(self, dim, precision=None, silent=False):
        self.dim = dim
        if precision is not None:
            self.precision = normalize(precision)
        else:
            self.precision = normalize(generate_random_matrix(self.dim))

        # Sanity checks before model build

        # assert np.all(np.linalg.eig(self.precision )[0] > 0.)
        assert self.precision.shape == (self.dim, self.dim)
        assert np.max(np.abs(self.precision -self.precision.T)) < 1e-8
        assert np.mean(np.diag(np.linalg.inv(self.precision))) - 1. < 1e-8
        assert np.all(np.linalg.eigh(np.linalg.inv(self.precision))[0]) - 1. < 1e-8

        self.covariance = np.linalg.inv(self.precision)

        if not silent:
            print("precision matrix : ", self.precision, '\n\n\n\n\n')
            print("covariance matrix : ", self.covariance)

    def sample(self, n_samples):
        # Draw from multivariate normal of the right covariance
        tmp = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.covariance, size=n_samples)
        assert tmp.shape == (n_samples, self.dim)
        # Rescale obtained samples to enforce sum(s_i**2) = N exactly
        norms = np.sum(tmp**2, axis=1)
        norms = np.sqrt(self.dim / norms)
        tmp *= norms.reshape(norms.shape + (1,))
        assert np.all(np.abs(np.sum(tmp**2, axis=1) - self.dim) < 1e-10)
        return tmp

    def get_empirical_C(self, n_samples):
        n_samples = int(n_samples)
        obs = self.sample(n_samples)
        assert obs.shape == (n_samples, self.dim)
        assert np.dot(obs.T, obs).shape == (self.dim, self.dim)
        return np.dot(obs.T, obs) / n_samples
