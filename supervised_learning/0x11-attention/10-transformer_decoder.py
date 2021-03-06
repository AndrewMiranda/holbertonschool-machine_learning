#!/usr/bin/env python3
"""
    Attention
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """  create encoder for transformer """

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
            N - the number of blocks in the encoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layer
            target_vocab - the size of the target vocabulary
            max_seq_len - the maximum sequence length possible
            drop_rate - the dropout rate
            Sets the following public instance attributes:
            N - the number of blocks in the encoder
            dm - the dimensionality of the model
            embedding - the embedding layer for the inputs
            positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
                containing the positional encodings
            blocks - a list of length N containing all of the EncoderBlock‘s
            dropout - the dropout layer, applied to the positional encodings
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
            x - a tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder
            encoder_output - a tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training - a boolean to determine if the model is training
            look_ahead_mask - mask applied to 1st multi head attention layer
            padding_mask - mask to be applied to 2nd multi head attention layer
            Returns: tensor of shape
                (batch, target_seq_len, dm) containing decoder output
        """
        seq_len = x.get_shape().as_list()[1]
        # seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x = x * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = x + self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

            # attention_weights[f'decoder_layer{i+1}_block1'] = block1
            # attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x  # , attention_weights
