# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

current_dir = os.getcwd()

utils_path = os.path.join(current_dir, 'src', 'utils')
data_processing_path = os.path.join(current_dir, 'src', 'data_processing')

sys.path.append(utils_path)
sys.path.append(data_processing_path)

from group_manipulation import select_valid_outfits
from product_processing import get_des_product_class
from group_manipulation import create_configurations
from group_manipulation import create_combinations
from setup_utilities import load_config

FEATURE_NAME = 'des_product_class'
CODE_SETS = 'cod_outfit'
NAME_SET = 'outfit'

def build_valid_outfits_df():
    config = load_config()
    df_outfits = pd.read_csv(config['data']['outfits_path'])
    df_products = pd.read_csv(config['data']['products_path'])

    df_outfit_products = pd.merge(df_outfits, df_products, on='cod_modelo_color', how='outer')
	
    df_outfit_products['des_product_class'] = df_outfit_products.apply(get_des_product_class, axis=1)
    
    configurations_base = config['data']['configurations_base']
    optional_products = config['data']['optional_products']

    optional_configurations = create_combinations(optional_products, root=False)
    all_configurations = create_configurations(configurations_base, optional_products)

    selected_outfits_df, _ = select_valid_outfits(df_outfit_products, FEATURE_NAME, CODE_SETS, NAME_SET, all_configurations)
    selected_outfits_df['des_filename'] = selected_outfits_df['des_filename'].str.replace('datathon', './data')

    selected_outfits_df.to_csv(config['data']['valid_outfits_path'])

if __name__ == "__main__":
    build_valid_outfits_df()

