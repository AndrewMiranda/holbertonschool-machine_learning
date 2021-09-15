#!/usr/bin/env python3
"""Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function that performs a convolution with custom padding
    Args:
        images : is a numpy.ndarray with shape (m, h, w)
                containing multiple grayscale images
            m : is the number of images
            h : is the height in pixels of the images
            w : is the width in pixels of the images
        kernel : is a numpy.ndarray with shape (kh, kw) containing
                the kernel for the convolution
            kh : is the height of the kernel
            kw : is the width of the kernel
        padding is a tuple of (ph, pw)|
            ph : is the padding for the height of the image
            pw : is the padding for the width of the image
    Returns
        a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_num = np.arange(m)
    ph = padding[0]
    pw = padding[1]

    # pad images before convolution, padding always symmetric here
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')
    # output size depends on filter size and padding
    output = np.zeros(shape=(m,
                             h - kh + 1 + 2 * ph,
                             w - kw + 1 + 2 * pw))

    for i in range(h - kh + 1 + 2 * ph):
        for j in range(w - kw + 1 + 2 * pw):
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
