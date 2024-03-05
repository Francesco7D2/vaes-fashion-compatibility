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
from train_utilities import log_normal_pdf, compute_loss, train_step, generate_and_save_images
from image_processing import _bytes_feature, parse_tfrecord_fn
from vae256 import Conv_VAE

def train_outfit_embeddings(): 
    config = load_config()
    
    model = Conv_VAE()
    model.compile()

    epochs = config['training']['epochs']
    train_dataset_path = config['training']['train_dataset_path']
    train_sample_path = config['training']['train_sample_path']
    epoch_images_path = config['training']['epoch_images_path']
    model_weights_path = config['training']['model_weights_path']
    model_data_path = config['training']['model_data_path']
    
    raw_dataset = tf.data.TFRecordDataset(train_dataset_path)
    train_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    raw_dataset = tf.data.TFRecordDataset(train_sample_path)
    train_sample_dataset = raw_dataset.map(parse_tfrecord_fn)
    train_sample = train_sample_dataset.take(1)

    ELBO_list = []

    generate_and_save_images(model, 0, train_sample, epoch_images_path)

    for epoch in range(1, epochs + 1):
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
        display.clear_output(wait=False)
        generate_and_save_images(model, epoch, train_sample, epoch_images_path)
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

