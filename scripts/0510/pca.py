import numpy as np
import seaborn as sns
import pandas as pd

from gaussian import get_samples


def get_principal_axis(samples):
    """Get `n` principal axis.

    Args:
        samples: samples which have the shape of (n, d).
    Returns:
        pa: 1st princical axis of given samples.

    """
    # get a number of samples
    n = samples.shape[0]

    # shift samples by the mean
    samples = samples - np.mean(samples, axis=0)

    # calculate the covariance matrix
    var = np.dot(samples.T, samples) / n

    # get the eigenvalues and the eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(var)

    # extract 1st element of eigenvectors and normalize it
    pa = eigenvectors[0]

    # normalize the vector by the norm
    pa /= np.linalg.norm(pa)

    return pa


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
    pa_angle = pa[0] / pa[1]

    # plot samples to 2d-plane
    data = np.concatenate((samples, labels), axis=1)
    data = pd.DataFrame(data, columns=['x', 'y', 'dist_id'])
    g = sns.lmplot('x', 'y', data=data, hue='dist_id', fit_reg=False)

    # plot the 1st principal axis
    x = np.linspace(data['x'].min(), data['x'].max())
    y = x * pa_angle
    sns.plt.plot(x, y, 'r')

    # show a graph
    sns.plt.show()


if __name__ == '__main__':
    main()
