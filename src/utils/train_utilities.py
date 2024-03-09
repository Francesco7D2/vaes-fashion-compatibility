# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
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
    
     
