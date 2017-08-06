"""Implementation of the VB algorithm."""
import argparse
from os.path import join
import pandas as pd

import numpy as np
from scipy.special import digamma

from utils import set_random_seed
from utils import init_logger


class VBALgorithm(object):
    """VB algorithm."""
    def __init__(self, k, duration, logger=None):
        """Initialize VBAlgorithm.

        Args:
            k: Number of Gaussian to be mixed.
            duration: Maximum number of iterations.
            logger: Logger.

        """
        self.k = k
        self.duration = duration
        self.logger = logger

        self.alpha = None
        self.beta = None
        self.v = None
        self.w = None
        self.m = None

    def fit(self, x):
        """Run VB algorithm."""
        n, d = x.shape

        alpha_0 = 1e-2
        beta_0 = 10.0
        v_0 = d
        w_0 = np.diag(np.ones(d))
        m_0 = np.random.rand(d)

        alpha = np.full((self.k,), self.k + alpha_0 + n / self.k)
        beta = np.full((self.k,), self.k + beta_0 + n / self.k)
        v = np.full((self.k,), v_0)
        w = np.zeros((self.k, d, d))
        for _k in range(self.k):
            w[_k] = np.diag(np.ones(d))
        m = np.random.rand(self.k, d)

        for t in range(self.duration):
            if self.logger:
                logger.info('Start {}-th iteration.'.format(t))

            # e step
            gamma = np.zeros((n, self.k))
            for _k in range(self.k):
                ln_lam = d * np.log(2) * np.linalg.det(w[_k])
                for _d in range(d):
                    ln_lam += digamma((v[_k] + 1 - _d) / 2)

                ln_pi = digamma(alpha[_k]) - digamma(alpha.sum())

                tmp = np.matmul(x - m[_k], w[_k])
                tmp = np.sum(np.multiply(tmp, x - m[_k]), axis=1)

                gamma[:, _k] = ln_pi + ln_lam / 2 - \
                    d / (2 * beta[_k]) - v[_k] * tmp / 2
                gamma[:, _k] = np.exp(gamma[:, _k])
            else:
                gamma = gamma / gamma.sum(axis=1, keepdims=True)

            # m step
            for _k in range(self.k):
                nk = gamma[:, _k].sum()

                mean_k = np.multiply(x.T, gamma[:, _k]).T
                mean_k = mean_k.sum(axis=0) / nk

                tmp = x - mean_k
                cov_k = np.dot(gamma[:, _k] * tmp.T, tmp) / nk

                alpha[_k] = alpha_0 + nk
                beta[_k] = beta_0 + nk
                m[_k] = (beta_0 * m_0 + nk * mean_k) / beta[_k]
                w[_k] = np.linalg.inv(w_0) + nk * cov_k + \
                    np.matmul(mean_k - m_0, (mean_k - m_0).T) * \
                    (beta_0 * nk) / (beta_0 + nk)
                w[_k] = np.linalg.inv(w[_k])
                v[_k] = v_0 + nk

        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.w = w
        self.m = m

        return gamma

    def params(self):
        """Returns params."""
        return (self.alpha, self.beta, self.v, self.w, self.m)


def main(args):
    """Entry point.

    Args:
        args: Command line arguemnts.

    """
    # load data
    x = pd.read_table(args.input, sep=',', header=None)
    x = x.as_matrix()

    # make an instance of VBAlgorithm
    vb = VBALgorithm(args.k, args.duration, logger)

    # run VB algorithm
    gamma = vb.fit(x)

    # get params
    alpha, beta, v, w, m = vb.params()

    # save parameters
    with open(join(args.output, 'params.vb.dat'), 'w') as f:
        f.write('* Alpha\n')
        for k, i in enumerate(alpha):
            f.write('** {}-th alpha\n'.format(k))
            f.write('{}\n'.format(i))
        f.write('* Beta\n')
        for k, i in enumerate(beta):
            f.write('** {}-th beta\n'.format(k))
            f.write('{}\n'.format(i))
        f.write('* V\n')
        for k, i in enumerate(v):
            f.write('** {}-th v\n'.format(k))
            f.write('{}\n'.format(i))
        f.write('* W\n')
        for k, i in enumerate(w):
            f.write('** {}-th w\n'.format(k))
            f.write('{}\n'.format(i))
        f.write('* M\n')
        for k, i in enumerate(m):
            f.write('** {}-th m\n'.format(k))
            f.write('{}\n'.format(i))

    # save gamma
    np.savetxt(join(args.output, 'z.vb.csv'), gamma, fmt='%.6f', delimiter=",")


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

    logger = init_logger('VB_Algorithm')
    logger.info('* k: {}'.format(args.k))
    logger.info('* input filename: {}'.format(args.input))
    logger.info('* output directory: {}'.format(args.output))

    main(args)
