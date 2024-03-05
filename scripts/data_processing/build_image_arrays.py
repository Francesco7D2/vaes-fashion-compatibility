# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import pickle
import tensorflow as tf
from  tqdm import tqdm

current_dir = os.getcwd()
utils_path = os.path.join(current_dir, 'src', 'utils')
data_processing_path = os.path.join(current_dir, 'src', 'data_processing')

sys.path.append(utils_path)
sys.path.append(data_processing_path)

from setup_utilities import load_config
from image_processing import load_and_preprocess_image
from image_processing import create_white_image




def build_image_arrays(compact = False):
    config = load_config()
    df_valid_outfits = pd.read_csv(config['data']['valid_outfits_path'])
    df_outfits_w_products = pd.read_csv(config['data']['outfits_w_products_path'])
    width = config['data']['image_width']
    height = config['data']['image_height']
    outfit_ims = []
    for index, row in tqdm(df_outfits_w_products.iterrows(), total=len(df_outfits_w_products), desc = "Building image arrays"):
        top = row['Tops']
        bottom = row['Bottoms']
        outer = row['Outerwear']
        dress = row['Dresses, jumpsuits and Complete set']
        foot = row['Footwear']
        bags = row['Bags']
        glasses = row['Glasses']
        earring = row['Earrings']
        ring = row['Ring']
        necklace = row['Necklace']

        if top is not np.nan:
            top_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == top]['des_filename']

            top_image = load_and_preprocess_image(top_image_path, width, height )
        else:
            top_image = create_white_image(width, height)

        if bottom is not np.nan:
            bottom_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == bottom]['des_filename']
            bottom_image = load_and_preprocess_image(bottom_image_path, width, height)
        else:
            bottom_image = create_white_image(width, height)

        if outer is not np.nan:
            outer_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == outer]['des_filename']
            outer_image = load_and_preprocess_image(outer_image_path, width, height)
        else:
            outer_image = create_white_image(width, height)

        if dress is not np.nan:
            dress_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == dress]['des_filename']
            dress_image = load_and_preprocess_image(dress_image_path, width, height)
        else:
            dress_image = create_white_image(width, height)


        if foot is not np.nan:
            foot_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == foot]['des_filename']
            foot_image = load_and_preprocess_image(foot_image_path, width, height)
        else:
            foot_image = create_white_image(width, height)

        if bags is not np.nan:
            bags_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == bags]['des_filename']
            bags_image = load_and_preprocess_image(bags_image_path, width, height)
        else:
            bags_image = create_white_image(width, height)

        if glasses is not np.nan:
            glasses_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == glasses]['des_filename']
            glasses_image = load_and_preprocess_image(glasses_image_path, width, height)
        else:
            glasses_image = create_white_image(width, height)


        if earring is not np.nan:
            earring_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == earring]['des_filename']
            earring_image = load_and_preprocess_image(earring_image_path, width, height)
        else:
            earring_image = create_white_image(width, height)

        if ring is not np.nan:
            ring_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == ring]['des_filename']
            ring_image = load_and_preprocess_image(ring_image_path, width, height)
        else:
            ring_image = create_white_image(width, height)

        if necklace is not np.nan:
            necklace_image_path = df_valid_outfits[df_valid_outfits['cod_modelo_color'] == necklace]['des_filename']
            necklace_image = load_and_preprocess_image(necklace_image_path, width, height)
        else:
            necklace_image = create_white_image(width, height)

        if compact:
            img = tf.concat([top_image, bottom_image, outer_image, dress_image,
                             foot_image, bags_image, glasses_image,
                             earring_image, ring_image, necklace_image], axis=3)
        else:
            img = tf.concat([top_image, bottom_image, outer_image, dress_image,
                             foot_image, bags_image, glasses_image,
                             earring_image, ring_image, necklace_image], axis=2)

        outfit_ims.append(img)
    
    with open(config['data']['outfit_image_arrays_path'], 'wb') as file:
        pickle.dump(outfit_ims, file)


if __name__ == "__main__":
	build_image_arrays()
