#!/usr/bin/env python3
'''Neural Style Transfer Module'''
import numpy as np
import tensorflow as tf


class NST():
    '''class to perform tasks for neural style transfer'''

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        '''class constructor
        Args:
            style_image - the image used as a style reference, stored as a
                numpy.ndarray
            content_image - the image used as a content reference, stored as a
                numpy.ndarray
            alpha - the weight for content cost
            beta - the weight for style cost
        '''

        if (type(style_image) is not np.ndarray or
                style_image.ndim != 3 or
                style_image.shape[2] != 3):
            raise TypeError("style_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        if (type(content_image) is not np.ndarray or
                content_image.ndim != 3 or
                content_image.shape[2] != 3):
            raise TypeError("content_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        if (type(alpha) not in [int, float] or alpha < 0):
            raise TypeError("alpha must be a non-negative number")

        if (type(beta) not in [int, float] or beta < 0):
            raise TypeError("beta must be a non-negative number")

        # enable eager execution:
        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        '''rescales an image such that its pixels values are between
               0 and 1 and its largest side is 512 pixels
        Args:
            image - a numpy.ndarray of shape (h, w, 3) containing the
                image to be scaled
        Returns: the scaled image
        '''
        if (type(image) is not np.ndarray or
                image.ndim != 3 or
                image.shape[2] != 3):
            raise TypeError("image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        # image scale:
        max_dim = 512
        h, w, _ = image.shape
        scale = max_dim / max(h, w)

        # batch dimension
        image = np.expand_dims(image, axis=0)

        # resize image
        image_t = tf.image.resize_bicubic(
            images=image,
            size=[int(h * scale), int(w * scale)],
        )

        # normalize image:
        image_t = image_t / 255
        image_t = tf.clip_by_value(image_t, 0, 1)

        return image_t

    def load_model(self):
        '''creates the model used to calculate cost'''

        style_outputs = []
        content_outputs = []

        vgg = tf.keras.applications.VGG19(
            include_top=False,
        )

        for i, layer in enumerate(vgg.layers):
            if (i == 0):
                output = vgg.input
                continue
            if (type(layer) is tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name,
                )
            output = layer(output)
            layer.trainable = False
            if (layer.name in self.style_layers):
                style_outputs.append(output)
            if (layer.name == self.content_layer):
                content_outputs.append(output)

        model_outputs = style_outputs + content_outputs

        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        '''calculates the gram matrix of a layer
        Args:
            input_layer - an instance of tf.Tensor or tf.Variable of
                shape (1, h, w, c)containing the layer output whose gram
                matrix should be calculated
        Returns: a tf.Tensor of shape (1, c, c) containing the gram matrix of
            input_layer
        '''
        a = input_layer

        if (not (isinstance(a, tf.Tensor) or isinstance(a, tf.Variable))
                or tf.rank(a).numpy() != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        # === way 1 ===
        # gram = tf.tensordot(a, a, [[0, 1, 2], [0, 1, 2]])

        # === way 2 ===
        # channels = int(a.shape[-1])
        # a_ = tf.reshape(a, [-1, channels])
        # gram = tf.matmul(a_, a_, transpose_a=True)

        # === way 3 - einstein notation ===
        # b: batch, h:height, w:width, c: channels, s: second_channels
        gram = tf.linalg.einsum("bhwc,bhws->bcs", a, a)

        # normalize gram matrix
        hw = tf.cast(a.shape[1] * a.shape[2], tf.float32)
        gram_normalized = gram / hw

        # for ways 1 & 2 - to recover the batch dimension
        # gram_normalized = tf.expand_dims(gram_normalized, axis=0)

        return gram_normalized

    def generate_features(self):
        '''extracts the features used to calculate neural style cost'''

        # Preprocess images
        # expects input in [0,1] so image is multiplied by 255
        p_style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255,
        )
        p_content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255,
        )

        n_style_layers = len(self.style_layers)
        style_features = self.model(p_style_image)[:n_style_layers]

        self.content_feature = self.model(p_content_image)[n_style_layers:][0]
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_features]

    def layer_style_cost(self, style_output, gram_target):
        '''Calculates the style cost for a single layer
        Args:
            style_output - tf.Tensor of shape (1, h, w, c) containing the layer
                style output of the generated image
            gram_target - tf.Tensor of shape (1, c, c) the gram matrix of the
                target style output for that layer
        Returns: the layer’s style cost
        '''
        if (not (isinstance(style_output, tf.Tensor) or
                 isinstance(style_output, tf.Variable)) or
                tf.rank(style_output).numpy() != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape.as_list()[-1]  # channels

        if (not (isinstance(gram_target, tf.Tensor) or
                 isinstance(gram_target, tf.Variable)) or
                tf.rank(gram_target).numpy() != 3 or
                gram_target.shape != (1, c, c)):
            raise TypeError("gram_target must be a tensor of "
                            "shape [1, {}, {}]".format(c, c))

        gram_style = self.gram_matrix(style_output)

        # === way 1 ===
        # return tf.reduce_mean(tf.square(gram_style - gram_target))

        # === way 2 ===
        return tf.reduce_sum(tf.square(gram_style - gram_target)) / (c ** 2)

    def style_cost(self, style_outputs):
        '''Calculates the style cost for generated image
        Args:
            style_outputs - a list of tf.Tensor style outputs for the
                generated image
        Returns: the style cost
        '''
        len_style_layers = len(self.style_layers)

        if (type(style_outputs) is not list or
                len(style_outputs) != len_style_layers):
            raise TypeError("style_outputs must be a list with a "
                            "length of {}".format(len_style_layers))

        weight = 1 / len_style_layers
        s_cost = 0

        gram_targets = self.gram_style_features

        for s_output, target in zip(style_outputs, self.gram_style_features):
            s_cost += self.layer_style_cost(s_output, target) * weight

        return s_cost

    def content_cost(self, content_output):
        '''Calculates the content cost for the generated image
        Args:
            content_output - a tf.Tensor containing the content output for
                the generated image
        Returns: the content cost
        '''
        content_shape = self.content_feature.shape

        if (not (isinstance(content_output, tf.Tensor) or
                 isinstance(content_output, tf.Variable)) or
                content_output.shape != content_shape):
            raise TypeError("content_output must be a tensor of "
                            "shape {}".format(content_shape))

        # === way 1 ===
        # cc = tf.reduce_mean(tf.square(content_output - self.content_feature))
        # return cc

        # === way 2 ===
        h, w, c = content_output.shape[1:]
        square = tf.square(content_output - self.content_feature)
        hwc = tf.cast(h * w * c, tf.float32)

        return tf.reduce_sum(square) / hwc

    def total_cost(self, generated_image):
        '''Calculates the total cost for the generated image
        Args:
            generated_image - a tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image
        Returns: (J, J_content, J_style)
                 - J is the total cost
                 - J_content is the content cost
                 - J_style is the style cost
        '''
        content_image_shape = self.content_image.shape

        if (not (isinstance(generated_image, tf.Tensor) or
                 isinstance(generated_image, tf.Variable)) or
                generated_image.shape != content_image_shape):
            raise TypeError("generated_image must be a tensor of "
                            "shape {}".format(content_image_shape))

        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )

        outputs = self.model(generated_image)

        style_outputs = outputs[:-1]
        style_cost = self.style_cost(style_outputs)

        content_output = outputs[-1]
        content_cost = self.content_cost(content_output)

        total_c = style_cost * self.beta + content_cost * self.alpha
        return (total_c, content_cost, style_cost)

    def compute_grads(self, generated_image):
        '''Calculates the gradients for the generated image
        Args:
            generated_image - a tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image
        Returns: gradients, J, J_content, J_style
                 - gradients is a tf.Tensor containing the gradients for the
                       generated image
                 - J is the total cost
                 - J_content is the content cost
                 - J_style is the style cost
        '''
        content_image_shape = self.content_image.shape

        if (not (isinstance(generated_image, tf.Tensor) or
                 isinstance(generated_image, tf.Variable)) or
                generated_image.shape != content_image_shape):
            raise TypeError("generated_image must be a tensor of "
                            "shape {}".format(content_image_shape))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            loss = self.total_cost(generated_image)

        gradients = tape.gradient(loss[0], generated_image)

        return gradients, loss[0], loss[1], loss[2]

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        '''generates the style transfered image
        Args:
            iterations - the number of iterations to perform gradient descent
                over
            step - if not None, the step at which you should print information
                about the training, including the final iteration:
                 - print Cost at iteration {i}: {J_total}, content {J_content},
                     style {J_style}
                     * i is the iteration
                     * J_total is the total cost
                     * J_content is the content cost
                     * J_style is the style cost
            lr - the learning rate for gradient descent
            beta1 - the beta1 parameter for gradient descent
            beta2 - the beta2 parameter for gradient descent
        Returns: generated_image, cost
            - generated_image is the best generated image
            - cost is the best cost
        '''
        if (type(iterations) is not int):
            raise TypeError("iterations must be an integer")
        if (iterations < 1):
            raise ValueError("iterations must be positive")
        if (step is not None and type(step) is not int):
            raise TypeError("step must be an integer")
        if (step is not None and (step < 1 or step > iterations)):
            raise ValueError("step must be positive and less than iterations")
        if (type(lr) not in [int, float]):
            raise TypeError("lr must be a number")
        if (lr <= 0):
            raise ValueError("lr must be positive")
        if (type(beta1) is not float):
            raise TypeError("beta1 must be a float")
        if (beta1 < 0 or beta1 > 1):
            raise ValueError("beta1 must be in the range [0, 1]")
        if (type(beta2) is not float):
            raise TypeError("beta2 must be a float")
        if (beta2 < 0 or beta2 > 1):
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.contrib.eager.Variable(self.content_image)

        opt = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
        )

        best_loss = tf.cast(0, tf.float32)

        for i in range(iterations + 1):
            # calculate gradients:
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image
            )

            # keep track of the best cost and the image associated with it
            if (J_total < best_loss or best_loss.numpy() == 0):
                best_loss = J_total
                best_img = generated_image

            # gradient descent;
            opt.apply_gradients([(grads, generated_image)])

            # step info:
            if (step is not None and (i % step == 0 or i == iterations)):
                print("Cost at iteration {}: {}, content {}, "
                      "style {}".format(i, J_total, J_content, J_style))

        best_img = np.squeeze(best_img.numpy(), 0)

        # ===== next lines were trying to depreprocess for VGG19 ====

        # best_img = tf.contrib.eager.Variable(self.content_image)
        # best_img = tf.keras.applications.vgg19.preprocess_input(
        #     best_img * 255
        # )
        # best_img = np.squeeze(best_img.numpy(), 0)
        # # perform the inverse of the preprocessing step
        # best_img[:, :, 0] += 103.939
        # best_img[:, :, 1] += 116.779
        # best_img[:, :, 2] += 123.68
        # # 'BGR'->'RGB'
        # best_img = best_img[:, :, ::-1]
        # best_img = np.clip(best_img, 0, 255).astype('uint8')

        # ============================================================

        return best_img, best_loss.numpy()

    def variational_cost(generated_image):
        '''Calculates the variational cost for the generated image'''
        return None
