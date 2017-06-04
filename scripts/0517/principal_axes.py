import numpy as np
from PIL import Image

from ccr import load_image, get_image_list, pca


def visualize_principal_axes(pa, top=3):
    """Visualize the calculated top three principal axes.

    Args:
        pa: principal axes.

    """
    img_width = np.int(np.sqrt(pa.shape[1]))
    img_height = img_width

    for i in range(top):
        img = convert_array_to_image(pa[i], img_width, img_height)
        img.save('results/top_{}_pa.png'.format(i+1))


def convert_array_to_image(array, width, height):
    """Reshape a numpy array to an image.

    Args:
        array: a numpy array.
        width: width of a image.
        height: height of a image.
    Returns:
        img: an image whose type is PIL.Image.

    """
    # change the value range to 0-255
    array = 255 * ((array - np.min(array)) / (np.max(array) - np.min(array)))

    # reshape the array
    img = np.reshape(array, (height, width))

    # convert to an Image onject
    img = Image.fromarray(img)
    img = img.convert('RGB')

    return img


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

    # visualize the principal axes
    visualize_principal_axes(pa, top=3)


if __name__ == '__main__':
    main()
