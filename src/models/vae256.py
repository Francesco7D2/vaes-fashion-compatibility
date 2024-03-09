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
    def call(self, inputs, training=None, mask=None):
      x = self.encoder(inputs)
      mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
      z = self.reparameterize(mean, logvar)
      reconstructed = self.decoder(z)
      return reconstructed



def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def compute_loss(model, x):
    """Computes the loss for a given input batch."""
    x_logit = model(x)
    # Cast the labels to float32 to match the type of logits
    x = tf.cast(x, dtype=tf.float32)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    return tf.reduce_mean(cross_ent)


def generate_and_save_images(model, epoch, test_sample, path):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(10, 10))
    for i, (img_pred, img_real) in enumerate(zip(predictions, test_sample)):
        plt.subplot(8, 1, 2*i + 1)
        plt.imshow(img_real)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.subplot(8, 1, 2*i + 2)
        plt.imshow(img_pred)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.01, wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    
     
