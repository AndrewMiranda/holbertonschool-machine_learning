#!/usr/bin/env python3
"""Forward Pooling"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs forward propagation over a pooling layer
    Args:
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            m: is the number of examples
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
            c_prev: is the number of channels in the previous layer
        kernel_shape: is a tuple of (kh, kw) containing the size of the
                kernel for the pooling
            kh is the kernel height
            kw is the kernel width
        stride: is a tuple of (sh, sw) containing the strides
                for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode: is a string containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively
    Returns:
        the output of the convolutional laye
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]
    func = {'max': np.max, 'avg': np.mean}

    output = np.zeros(shape=(m,
                             int((h_prev - kh) / sh + 1),
                             int((w_prev - kw) / sw + 1),
                             c_prev))
    if mode in ['max', 'avg']:
        for i in range(int((h_prev - kh) / sh + 1)):
            for j in range(int((w_prev - kw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j,
                    :
                ] = func[mode](
                    A_prev[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ], axis=(1, 2)
                )
    return output
