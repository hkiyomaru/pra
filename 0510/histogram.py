import numpy as np
import seaborn as sns
import pandas as pd

from gaussian import get_samples
from pca import get_principal_axis
from fisher import get_fisher_axis


def map_samples(samples, axis):
    """Map samples based on a given axis.

    Args:
        samples: samples which have the shape of (n, d).
        axis: 1-dimensional vector which have the shape of (d, ).
    """
    # map samples
    samples = np.inner(axis.T, samples)

    # reshape samples
    samples = np.reshape(samples, (-1, 1))

    return samples


def main():
    """Entry point."""
    # get samples that follow a Gaussian distribution (1)
    mean1 = (3, 1)
    var1 = ([1, 2], [2, 5])
    samples1, labels1 = get_samples(mean1, var1, dist_id=1)

    # get samples that follow a Gaussian distribution (2)
    mean2 = (1, 3)
    var2 = ([1, 2], [2, 5])
    samples2, labels2 = get_samples(mean2, var2, dist_id=2)

    # concat samples and labels respectively
    samples = np.concatenate((samples1, samples2), axis=0)
    labels = np.concatenate((labels1, labels2), axis=0)

    # get the 1st principal axis
    pa = get_principal_axis(samples)

    # get the fisher axis
    fa = get_fisher_axis(samples1, samples2)

    # map into 1d-plane based on `pa`
    pa_samples = map_samples(samples, pa)

    # map into 1d-plane based on `fa`
    fa_samples = map_samples(samples, fa)

    # plot samples to 2d-plane (pca)
    pa_data = np.concatenate((pa_samples, labels), axis=1)
    pa_data = pd.DataFrame(pa_data, columns=['z', 'dist_id'])
    g = sns.FacetGrid(pa_data, hue='dist_id', size=8)
    g.map(sns.plt.hist, 'z', alpha=0.8)

    # show a graph
    sns.plt.show()

    # plot samples to 2d-plane (pca)
    fa_data = np.concatenate((fa_samples, labels), axis=1)
    fa_data = pd.DataFrame(fa_data, columns=['z', 'dist_id'])
    g = sns.FacetGrid(fa_data, hue='dist_id', size=8)
    g.map(sns.plt.hist, 'z', alpha=0.8)

    # show a graph
    sns.plt.show()


if __name__ == '__main__':
    main()
