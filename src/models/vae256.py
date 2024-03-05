# -*- coding: utf-8 -*-

import tensorflow as tf


class Conv_VAE(tf.keras.Model):
    """Convolutional Variational autoencoder."""

    def __init__(self):
        super(Conv_VAE, self).__init__()
        self.latent_space = []
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 96*10, 3)),
                tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), padding="same"),
                tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), padding="same"),
                tf.keras.layers.MaxPooling2D(),

                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same"),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same"),
                tf.keras.layers.MaxPooling2D(),

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(512, activation = tf.nn.relu),
                tf.keras.layers.Dense(256, activation = tf.nn.relu),
                tf.keras.layers.Dense(512, activation = tf.nn.relu),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(256,)),
                tf.keras.layers.Dense(units=32*240*8, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(32, 240, 8)),

                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same"),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same"),

                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same"),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same"),

                tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding="same"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
      if eps is None:
        eps = tf.random.normal(shape=(100, self.latent_dim))
      return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
      mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
      return mean, logvar

    def reparameterize(self, mean, logvar):
      eps = tf.random.normal(shape=mean.shape)
      return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
      logits = self.decoder(z)
      if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
      return logits


