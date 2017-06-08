from os import listdir

import numpy as np
import seaborn as sns
from PIL import Image


def load_image(path_to_image):
    """Load an image as an numpy array.

    Args:
        path_to_image: path to an image.
    Returns:
        image: an image, where the type is numpy array.

    """
    # load an image using Image module.
    image = Image.open(path_to_image)

    # change the type.
    image = np.asarray(image)
    image = image.flatten()

    return image


def get_image_list(path_to_image_dir):
    """Get an image list.

    Args:
        path_to_image_dir: path to a directory where images have been saved.
    Returns:
        image_list: A list which includes paths to images.

    """
    files = [f for f in listdir(path_to_image_dir)]

    return files


def pca(mat):
    """Apply PCA to given images.

    Args:
        mat: images represented by numpy arrays.
    Returns:
        pa: principal axes.
        ev: eigenvalues which are corresponding to the principal axes.
        r: rank of given matrix.

    """
    # get a shape of given matrix
    n, m = mat.shape

    # get the rank of the matrix
    r = min(n, m)

    # shift the values by the mean
    mat = mat - np.mean(mat, axis=0)

    # calculate eigenvectors and eigenvalues for `np.dot(mat, mat.T) / n`
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(mat, mat.T) / n)

    # get the principal axes
    pa = np.zeros((r, m))
    ev = np.zeros((r,))
    for i in range(r):
        pa[i] = np.dot(mat.T, eigenvectors[i])
        pa[i] /= np.linalg.norm(pa[i])  # normalize eigenvectors
        ev[i] = eigenvalues[i]

    return pa, ev, r


def main():
    """Entry point."""
    # get an image list
    base_url = '../../data/0517/'
    image_list = get_image_list(base_url)

    # load the images
    images = [load_image(base_url + filename) for filename in image_list]
    images = np.asarray(images)

    # get principal axes
    pa, ev, r = pca(images)

    # calculate CCR
    contrib = ev / np.sum(ev)
    ccr = np.zeros_like(contrib)
    for i in range(r):
        ccr[i] = np.sum(contrib[:i])

    # visualize CCR
    sns.plt.plot(range(r), ccr)
    sns.plt.xlabel('number of dimensions')
    sns.plt.ylabel('cumulative contribution ratio (CCR)')
    sns.plt.savefig('results/ccr.png')


if __name__ == '__main__':
    main()
