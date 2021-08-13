#!/usr/bin/env python3
"""Module for the function train
"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train,
          Y_train,
          X_valid,
          Y_valid,
          layer_sizes,
          activations,
          alpha,
          iterations,
          save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier
    Args:
        X_train: A numpy.ndarray containing the training input data
        Y_train: A numpy.ndarray containing the training labels
        X_valid: A numpy.ndarray containing the validation input data
        Y_valid: A numpy.ndarray containing the validation labels
        layer_sizes: A list containing the number of nodes in each layer of
        the network
        activations:A list containing the activation functions for each layer
        of the network
        alpha: The learning rate
        iterations: The number of iterations to train over
        save_path: Designates where to save the model
    Returns:
        The path where the model was saved
    """

    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    acc = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', acc)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    op = create_train_op(loss, alpha)
    tf.add_to_collection('op', op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            t_acc, t_loss = sess.run([acc, loss], {x: X_train, y: Y_train})
            v_acc, v_loss = sess.run([acc, loss], {x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_loss))
                print("\tTraining Accuracy: {}".format(t_acc))
                print("\tValidation Cost: {}".format(v_loss))
                print("\tValidation Accuracy: {}".format(v_acc))
            if i < iterations:
                sess.run(op, {x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

    return save_path
