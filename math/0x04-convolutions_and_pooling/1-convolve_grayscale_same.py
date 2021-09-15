#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a valid convolution
        on grayscale images
    Args:
        images : is a numpy.ndarray with shape (m, h, w)
                containing multiple grayscale images
            m : is the number of images
            h : is the height in pixels of the images
            w : is the width in pixels of the images
        kernel : is a numpy.ndarray with shape (kh, kw) containing
                the kernel for the convolution
            kh : is the height of the kernel
            kw : is the width of the kernel|
    Returns
        a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_num = np.arange(m)
    output = np.zeros(shape=(m, h, w))

    # pad images before convolution
    # handle even vs. odd filter sizes with np.ceil()
    ph = int(np.ceil((kh - 1)/2))
    pw = int(np.ceil((kw - 1)/2))

    # pad images accordingly, padding always symmetric here
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    for i in range(h):
        for j in range(w):
            output[
                image_num,
                i,
                j
            ] = np.sum(
                padded_images[
                    image_num,
                    i: i + kh,
                    j: j + kw
                ] * kernel,
                axis=(1, 2)
            )
    return output
