import numpy as np


def generate_random_matrix(dim):
    sigma = np.random.normal(np.zeros([dim,dim], dtype=np.float32), 1./np.sqrt(dim))
    sigma = (sigma + sigma.T) / np.sqrt(2)
    return sigma


def normalize(precision_matrix):
    # Impose PD and diagonal of inverse equal 1 -> covariance of all x equal to 1
    # The diagonal of precision matrix is assumed 0
    dim = precision_matrix.shape[0]

    precision_matrix -= np.diag(np.diag(precision_matrix))
    offset = np.abs(np.min(np.linalg.eig(precision_matrix)[0])) + 0.01
    precision_matrix += np.diag([offset for _ in range(dim)])
    precision_matrix *= np.diag(np.linalg.inv(precision_matrix))[0]

    return precision_matrix


class CenteredGM:
    def __init__(self, dim, precision=None):
        self.dim = dim
        if precision is not None:
            self.precision = normalize(precision)
        else:
            self.precision = normalize(generate_random_matrix(self.dim))

        # Sanity checks before model build

        assert np.all(np.linalg.eig(self.precision )[0] > 0.)
        assert self.precision.shape == (self.dim, self.dim)
        assert np.max(np.abs(self.precision -self.precision.T)) < 1e-8
        assert np.all(np.abs(np.diag(np.linalg.inv(self.precision)) - 1) < 1e-8)

        print("precision matrix : ", self.precision, '\n\n\n\n\n')

        self.covariance = np.linalg.inv(self.precision)
        print("covariance matrix : ", self.covariance)

    def sample(self, n_samples):
        return np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.covariance, size=int(n_samples))

    def get_empirical_C(self, n_samples):
        obs = self.sample(n_samples)
        assert obs.shape == (n_samples, self.dim)
        assert np.dot(obs.T, obs).shape == (self.dim, self.dim)
        return np.dot(obs.T, obs) / n_samples
