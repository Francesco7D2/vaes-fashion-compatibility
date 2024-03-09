# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import pickle
import tensorflow as tf
from tqdm import tqdm
import time

current_dir = os.getcwd()
utils_path = os.path.join(current_dir, 'src', 'utils')
data_processing_path = os.path.join(current_dir, 'src', 'data_processing')
models_path = os.path.join(current_dir, 'src', 'models')

sys.path.append(utils_path)
sys.path.append(data_processing_path)
sys.path.append(models_path)

from setup_utilities import load_config
from image_processing import _bytes_feature, parse_tfrecord_fn
#from vae256 import Conv_VAE

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
    def call(self, inputs, training=None, mask=None):
      x = self.encoder(inputs)
      mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
      z = self.reparameterize(mean, logvar)
      reconstructed = self.decoder(z)
      return reconstructed

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
    
    

def compute_loss(model, x):
    """Computes the loss for a given input batch."""
    

    # Cast the labels to float32 to match the type of logits
    x = tf.cast(x, dtype=tf.float32)
    
    x_logit = model(x)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    return tf.reduce_mean(cross_ent)



def train_outfit_embeddings(): 
    config = load_config()
    
    model = Conv_VAE()
    
    
    dummy_input = tf.ones((1, 128, 96 * 10, 3))


    model.build(dummy_input.shape)


    print(model.summary())
    
    #model.layers[0].build(input_shape=(None, 512))

    #model.layers[1].build(input_shape=(None, 128, 960, 3))
    
    model.compile(optimizer = tf.keras.optimizers.Adam(1e-4))
    
    print("Summary for the first submodel:")
    print(model.layers[0].summary())

    print("Summary for the second submodel:")
    print(model.layers[1].summary())

    epochs = config['training']['epochs']
    train_dataset_path = config['training']['train_dataset_path']
    train_sample_path = config['training']['train_sample_path']
    epoch_images_path = config['training']['epoch_images_path']
    model_weights_path = config['training']['model_weights_path']
    model_data_path = config['training']['model_data_path']
    
    raw_dataset = tf.data.TFRecordDataset(train_dataset_path)
    train_dataset = raw_dataset.map(parse_tfrecord_fn)
    train_dataset = train_dataset.map(lambda x: tf.expand_dims(x, axis=0))


    raw_dataset = tf.data.TFRecordDataset(train_sample_path)
    train_sample_dataset = raw_dataset.map(parse_tfrecord_fn)

    
    optimizer = tf.keras.optimizers.Adam(1e-4)

    ELBO_list = []

    #generate_and_save_images(model, 0, train_sample, epoch_images_path)

    for epoch in tqdm(range(1, epochs + 1), total=epochs):
        start_time = time.time()
        latent_space_list = []
        for train_x in tqdm(train_dataset, desc=f'train step for epoch: {epoch}'):
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for train_x in tqdm(train_dataset, desc=f'compute loss (train) for epoch: {epoch}'):
            loss(compute_loss(model, train_x))
        elbo = -loss.result()
        ELBO_list.append(elbo)

        print('Epoch: {}, Train set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        #display.clear_output(wait=False)
        #generate_and_save_images(model, epoch, train_sample, epoch_images_path)
        if epoch % 20 == 0:
            model.save_weights(model_weights_path)

            model.latent_space = []
            for train_x in train_dataset:
                mean, logvar = model.encode(train_x)
                z = model.reparameterize(mean, logvar)
                model.latent_space.append(z)

            with open(model_data_path, 'wb') as file:
                data_to_save = {'latent_space': model.latent_space, 'ELBO_list': ELBO_list}
                pickle.dump(data_to_save, file)

if __name__ == "__main__":
    train_outfit_embeddings()

