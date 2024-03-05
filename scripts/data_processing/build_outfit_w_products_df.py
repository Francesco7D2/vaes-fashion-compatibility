# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

current_dir = os.getcwd()
utils_path = os.path.join(current_dir, 'src', 'utils')
sys.path.append(utils_path)

from setup_utilities import load_config


CODE_SET = 'cod_outfit'
CODE_PRODUCT = 'cod_modelo_color'
FEATURE_NAME = 'des_product_class' 

def build_outfit_w_products():
    config = load_config()
    df = pd.read_csv(config['data']['valid_outfits_path'])

    df = df.groupby(CODE_SET).agg({
        FEATURE_NAME: list,
        CODE_PRODUCT: list,
    })

    df_outfits = []
    for index, row in df.iterrows():
        product_name_list = row['des_product_class']
        product_code_list = row['cod_modelo_color']

        outfit_dict = {'Tops': np.nan, 'Bottoms': np.nan, 'Outerwear': np.nan, 'Dresses, jumpsuits and Complete set': np.nan,
                       'Footwear': np.nan, 'Bags': np.nan, 'Glasses': np.nan, 'Earrings': np.nan, 'Ring': np.nan, 'Necklace': np.nan}

        for name_product in product_name_list:
            idx_prod = product_name_list.index(name_product)
            if name_product == 'Tops':
                outfit_dict['Tops'] = product_code_list[idx_prod]
            elif name_product == 'Bottoms':
                outfit_dict['Bottoms'] = product_code_list[idx_prod]
            elif name_product == 'Outerwear':
                outfit_dict['Outerwear'] = product_code_list[idx_prod]
            elif name_product == 'Dresses, jumpsuits and Complete set':
                outfit_dict['Dresses, jumpsuits and Complete set'] = product_code_list[idx_prod]
            elif name_product == 'Footwear':
                outfit_dict['Footwear'] = product_code_list[idx_prod]
            elif name_product == 'Bags':
                outfit_dict['Bags'] = product_code_list[idx_prod]
            elif name_product == 'Glasses':
                outfit_dict['Glasses'] = product_code_list[idx_prod]
            elif name_product == 'Earrings':
                outfit_dict['Earrings'] = product_code_list[idx_prod]
            elif name_product == 'Ring':
                outfit_dict['Ring'] = product_code_list[idx_prod]
            elif name_product == 'Necklace':
                outfit_dict['Necklace'] = product_code_list[idx_prod]
        outfit_dict[CODE_SET] = index
        df_outfits.append(outfit_dict)

    df_outfits = pd.DataFrame(df_outfits)
    df_outfits.to_csv(config['data']['outfits_w_products_path'], index=False)



if __name__ == "__main__":
    build_outfit_w_products()

