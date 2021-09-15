#!/usr/bin/env python3
"""Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """function that performs pooling on images
    Args:
        images : is a numpy.ndarray with shape (m, h, w, c)
                containing multiple grayscale images
            m : is the number of images
            h : is the height in pixels of the images
            w : is the width in pixels of the images
            c : is the number of channels in the image
        kernel_shape : is a numpy.ndarray with shape (kh, kw) containing
                the kernel for the convolution
            kh : is the height of the kernel
            kw : is the width of the kernel
        stride is a tuple of (sh, sw)
            sh : is the stride for the height of the image
            sw : is the stride for the width of the image
        mode indicates the type of pooling
            max : indicates max pooling
            avg : indicates average pooling
    Returns
        a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]
    func = {'max': np.max, 'avg': np.mean}

    output = np.zeros(shape=(m,
                             int((h - kh) / sh + 1),
                             int((w - kw) / sw + 1),
                             c))
    if mode in ['max', 'avg']:
        for i in range(int((h - kh) / sh + 1)):
            for j in range(int((w - kw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j,
                    :
                ] = func[mode](
                    images[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ], axis=(1, 2)
                )
    return output
