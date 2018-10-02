import numpy as np


def generate_pd_matrix(dim):
    sigma = np.random.normal(np.zeros([dim,dim], dtype=np.float32), 1./np.sqrt(dim))
    sigma = (sigma + sigma.T) / np.sqrt(2)
    return sigma


def normalize(sigma):
    dim = sigma.shape[0]
    # Impose PD and mean of eigenvalues equal 1

    # Forces all diagonal elements to be the same
    # That's the easiest way to ensure all ev > 0...
    sigma -= np.diag(np.diag(sigma))
    sigma += np.diag([dim for _ in range(dim)])

    sigma *= dim / np.sum(np.linalg.eig(sigma)[0])
    assert np.all(np.linalg.eig(sigma)[0] > 0.)
    assert np.abs(np.sum(np.linalg.eig(sigma)[0]) - dim) < 0.05

    return sigma


class CenteredGM:
    def __init__(self, dim, sigma=None):
        self.dim = dim
        if sigma is not None:
            self.sigma = normalize(sigma)
        else:
            self.sigma = normalize(generate_pd_matrix(self.dim))

        # Sanity checks before model build
        # print(np.linalg.eig(self.sigma)[0])
        assert np.all(np.linalg.eig(self.sigma)[0] > 0.)
        assert self.sigma.shape == (self.dim, self.dim)
        assert np.max(np.abs(self.sigma-self.sigma.T)) < 10**(-6)
        # assert np.max(np.abs(np.diag(self.sigma))) < 10**(-6)

    def sample(self, n_samples):
        return np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.sigma, size=int(n_samples))

    def get_empirical_C(self, n_samples):
        obs = self.sample(n_samples)
        assert obs.shape == (n_samples, self.dim)
        assert np.dot(obs.T, obs).shape == (self.dim, self.dim)
        return np.dot(obs.T, obs) / n_samples
