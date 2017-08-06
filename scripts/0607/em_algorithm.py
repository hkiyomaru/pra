"""Implementation of the EM algorithm."""
import argparse
from os.path import join
import pandas as pd

import numpy as np
from scipy.stats import multivariate_normal

from utils import set_random_seed
from utils import init_logger


class EMALgorithm(object):
    """EM algorithm."""
    def __init__(self, k, duration, logger=None):
        """Initialize EMAlgorithm.

        Args:
            k: Number of Gaussian to be mixed.
            duration: Maximum number of iterations.
            logger: Logger.

        """
        # hyper-parameter
        self.k = k

        # parameters
        self.mean = None
        self.cov = None
        self.pi = None

        # training configuration
        self.duration = duration
        self.logger = logger

    def gaussian(self, x, mean, cov):
        """Returns probability density in a normal distribution."""
        return multivariate_normal.pdf(x, mean=mean, cov=cov)

    def likelihood(self, x, mean, cov, pi):
        """Calculate likelihood."""
        l = 0
        for _k in range(self.k):
            l += self.gaussian(x, mean[_k], cov[_k]) * pi[_k]
        l = np.sum(np.log(l))

        return l

    def fit(self, x):
        """Run EM algorithm."""
        n, d = x.shape

        mean = np.random.rand(self.k, d)
        cov = np.zeros((self.k, d, d))
        for _k in range(self.k):
            cov[_k] = np.diag(np.ones(d))
        pi = np.random.rand(self.k)

        gamma = np.zeros((n, self.k))

        for t in range(self.duration):
            if self.logger:
                logger.info('Start {}-th iteration.'.format(t))

            # calculate likelihood
            l = self.likelihood(x, mean, cov, pi)

            # e step
            denom = np.zeros((n))
            for _k in range(self.k):
                denom += pi[_k] * self.gaussian(x, mean[_k], cov[_k])
            for _k in range(self.k):
                gamma[:, _k] = pi[_k] * \
                    self.gaussian(x, mean[_k], cov[_k]) / denom

            # m step
            for _k in range(self.k):
                nk = np.sum(gamma[:, _k])

                for _d in range(d):
                    mean[_k, _d] = np.sum(gamma[:, _k] * x[:, _d]) / nk

                tmp = x - mean[_k]
                cov[_k] = np.dot(gamma[:, _k] * tmp.T, tmp) / nk

                pi[_k] = nk / n

            _l = self.likelihood(x, mean, cov, pi)

            if self.logger:
                self.logger.info('Log likelihood: {0:.2f}'.format(_l))
                self.logger.info('Increased: {0:.2f}'.format(_l - l))

            if np.abs(_l - l) < 1e-2:
                if self.logger:
                    logger.info('The training has reached saturation.')
                break

        self.mean = mean
        self.cov = cov
        self.pi = pi

        return gamma

    def params(self):
        """Returns params."""
        return (self.mean, self.cov, self.pi)


def main(args):
    """Entry point.

    Args:
        args: Command line arguemnts.

    """
    # load data
    x = pd.read_table(args.input, sep=',', header=None)
    x = x.as_matrix()

    # make an instance of EMAlgorithm
    em = EMALgorithm(args.k, args.duration, logger)

    # run EM algorithm
    gamma = em.fit(x)

    # get params
    mean, cov, pi = em.params()

    # save parameters
    with open(join(args.output, 'params.em.dat'), 'w') as f:
        f.write('* Mean\n')
        for k, i in enumerate(mean):
            f.write('** {}-th mean\n'.format(k))
            f.write('{}\n'.format(i))
        f.write('* Variance\n')
        for k, i in enumerate(cov):
            f.write('** {}-th variance\n'.format(k))
            f.write('{}\n'.format(i))
        f.write('* Pi\n')
        for k, i in enumerate(pi):
            f.write('** {}-th pi\n'.format(k))
            f.write('{}\n'.format(i))

    # save gamma
    np.savetxt(join(args.output, 'z.em.csv'), gamma, fmt='%.6f', delimiter=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int,
                        help='number of gaussians to be mixed')
    parser.add_argument('--duration', type=int, default=10,
                        help='number of maximum iteration')
    parser.add_argument('-i', '--input', type=str,
                        help='path to input file')
    parser.add_argument('-o', '--output', type=str, default='.',
                        help='path to output directory')
    args = parser.parse_args()

    set_random_seed()

    logger = init_logger('EM_Algorithm')
    logger.info('* k: {}'.format(args.k))
    logger.info('* input filename: {}'.format(args.input))
    logger.info('* output directory: {}'.format(args.output))

    main(args)
