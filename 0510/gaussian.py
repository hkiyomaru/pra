import numpy as np
import pandas as pd
import seaborn as sns

# fix random seed
np.random.seed(0)


def get_samples(mean, var, dist_id, size=100):
    """Generate samples that follow a Gaussian distribution.

    Args:
        mean: mean of a Gaussian distribution.
        var: variance of a Gaussian distribution.
        dist_id: id for a distribution.
        size: number of samples.
    Returns:
        samples: samples generated from given distribution.
        labels: labels for samples.

    """
    samples = np.random.multivariate_normal(mean, var, size=size)
    labels = np.full((size, 1), dist_id)

    return samples, labels


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

    # plot samples to 2d-plane
    data = np.concatenate((samples, labels), axis=1)
    data = pd.DataFrame(data, columns=['x', 'y', 'dist_id'])
    sns.lmplot('x', 'y', data=data, hue='dist_id', fit_reg=False)

    # show the graph
    sns.plt.show()


if __name__ == '__main__':
    main()
