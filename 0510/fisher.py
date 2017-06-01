import numpy as np
import seaborn as sns
import pandas as pd

from gaussian import get_samples


def get_fisher_axis(samples1, samples2):
    """Get an axis that is calculated by Fisher LDA.

    Args:
        samples1: samples which have the shape of (n, d).
        samples2: samples which have the shape of (n, d).
    Returns:
        fa: an axis that is calculated by Fisher LDA.

    """
    # shift samples by the mean
    mean1 = np.mean(samples1, axis=0)
    mean2 = np.mean(samples2, axis=0)
    samples1 = samples1 - mean1
    samples2 = samples2 - mean2

    # calculate within-class scatter
    s1 = np.dot((samples1).T, (samples1))
    s2 = np.dot((samples2).T, (samples2))
    sw = s1 + s2

    # calculate the invertible metrix
    inv_sw = np.linalg.inv(sw)

    # get the fisher axis
    fa = np.matmul(inv_sw, mean1 - mean2)

    # normalize the vector by the norm
    fa /= np.linalg.norm(fa)

    return fa


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

    # get the fisher axis
    fa = get_fisher_axis(samples1, samples2)
    fa_angle = fa[0] / fa[1]

    # plot samples to 2d-plane
    data = np.concatenate((samples, labels), axis=1)
    data = pd.DataFrame(data, columns=['x', 'y', 'dist_id'])
    sns.lmplot('x', 'y', data=data, hue='dist_id', fit_reg=False)

    # plot the fisher axis
    x = np.linspace(data['x'].min(), data['x'].max())
    y = x * fa_angle
    sns.plt.plot(x, y, 'r')

    # show a graph
    sns.plt.show()


if __name__ == '__main__':
    main()
