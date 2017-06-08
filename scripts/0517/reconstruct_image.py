import numpy as np
from PIL import Image

from ccr import load_image, get_image_list, pca
from principal_axes import convert_array_to_image


def reconstruct(img, pa, img_id=None, top=3):
    """Reconstruct several images by using the principal axes.

    Args:
        img: an image represented by a numpy array.
        pa: principal axes.
        imd_id: image ID.

    """
    # apply PCA
    pa_to_use = pa[:top]
    img_pca = np.inner(pa_to_use, img)

    # reconstruct the image
    img_rec = np.inner(pa_to_use.T, img_pca)

    # show the reconstructed image (and also the original image)
    img_width = np.int(np.sqrt(pa.shape[1]))
    img_height = img_width
    img = convert_array_to_image(img, img_width, img_height)
    img_rec = convert_array_to_image(img_rec, img_width, img_height)

    canvas = Image.new('RGB', (img_width*2, img_height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.paste(img_rec, (img_width, 0))

    if img_id is None:
        canvas.show()
    else:
        canvas.save('results/rec_{}.png'.format(img_id))


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

    # reconstruct several images
    reconstruct(images[0], pa, img_id=0)  # scissors
    reconstruct(images[14], pa, img_id=14)  # rock
    reconstruct(images[28], pa, img_id=28)  # paper


if __name__ == '__main__':
    main()
