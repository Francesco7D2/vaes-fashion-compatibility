# -*- coding: utf-8 -*-




import pandas as pd

def get_des_product_class(row: pd.Series) -> pd.Series:
    """
    Extracts the desired product class based on specific conditions.

    Parameters:
    - row (pd.Series): The input row from a Pandas DataFrame containing
                      information about a product.

    Returns:
    - pd.Series: The extracted product class information.

    Conditions:
    - If 'des_product_category' is 'Accessories, Swim and Intimate' and
      'des_product_family' is not 'Jewellery', the function returns the
      'des_product_family'.
    - If 'des_product_family' is 'Jewellery', the function returns the
      'des_product_type'.
    - Otherwise, the function returns 'des_product_category'.
    """
    if row['des_product_category'] == 'Accesories, Swim and Intimate' and row['des_product_family'] != 'Jewellery':
        return row['des_product_family']
    elif row['des_product_family'] == 'Jewellery':
        return row['des_product_type']
    else:
        return row['des_product_category']
