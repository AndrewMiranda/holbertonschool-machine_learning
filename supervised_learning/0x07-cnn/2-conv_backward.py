#!/usr/bin/env python3
"""
Backpropagation over Convolution Layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function that performs a backpropagation over a convolutional layer
    Args:
        dZ: is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output of
            the convolutional layer
            m: is the number of examples
            h_new: is the height of the output
            w_new: is the width of the output
            c_new: is the number of channels in the output
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
            c_prev: is the number of channels in the previous layer
        W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
                containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
        b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        padding: is a string that is either same or valid, indicating the
                type of padding used
        stride: is a tuple of (sh, sw) containing the strides
                for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode: is a string containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively
    Returns:
         the partial derivatives with respect to the previous layer
         (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # output size depends on filter size and must be equal to image size
        # imposing constraints on padding for a given set of strides
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    if padding == 'same':
        # pad A_prev before convolution, padding always symmetric here
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    dA_prev = np.zeros(shape=A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.zeros(shape=b.shape)

    for img_num in range(m):
        for k in range(c_new):
            for i in range(h_new):
                for j in range(w_new):
                    dA_prev[
                        img_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ] += dZ[
                        img_num,
                        i,
                        j,
                        k
                    ] * W[:, :, :, k]
                    dW[:, :, :, k] += A_prev[
                        img_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ] * dZ[
                        img_num,
                        i,
                        j,
                        k
                    ]
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    if padding == 'same':
        dA_prev = dA_prev[
            :,
            ph: dA_prev.shape[1] - ph,
            pw: dA_prev.shape[2] - pw,
            :
        ]
    return dA_prev, dW, db
