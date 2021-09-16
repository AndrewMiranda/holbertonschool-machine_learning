#!/usr/bin/env python3
""" Deep Convolutional Neural Networks """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block as described in Densely Connected
      Convolutional Networks:
    Args:
        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        growth_rate is the growth rate for the dense block
        layers is the number of layers in the dense block
    """
    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    for i in range(layers):
        my_layer = K.layers.BatchNormalization()(X)
        my_layer = K.layers.Activation('relu')(my_layer)

        # conv 1×1 produces 4k (growth_rate) feature-maps
        my_layer = K.layers.Conv2D(filters=4*growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        my_layer = K.layers.BatchNormalization()(my_layer)
        my_layer = K.layers.Activation('relu')(my_layer)

        # conv 3×3 produces k (growth_rate) feature-maps
        my_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        X = K.layers.concatenate([X, my_layer])
        nb_filters += growth_rate

    return X, nb_filters
