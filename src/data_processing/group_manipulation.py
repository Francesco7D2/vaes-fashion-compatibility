# -*- coding: utf-8 -*-


import pandas as pd
from collections import Counter



def get_grouped_counts(df: pd.DataFrame, code_sets: str,name_set: str) -> pd.DataFrame:
    """
    Get grouped counts of a specified column in a DataFrame based on another 
    column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - code_sets (str): The column name to group by and count occurrences.
    - name_set (str): The column name to identify the sets and create the final 
    DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the grouped counts based on the 
    specified columns.
    """
    set_size = df.groupby(code_sets)[code_sets].count().reset_index(
        name=f'{name_set}_size'
    )
    grouped_set_size = set_size.groupby(f'{name_set}_size').size().reset_index(
        name=f'grouped_{name_set}_size'
    )
    return grouped_set_size



def get_grouped_counts_feature_values(df, code_sets, name_set, feature_name):
    """
    Groups DataFrame by specified code sets and aggregates count and list of
    feature values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing information.
    - code_sets (str or list): Column name or list of column names to group by.
    - name_set (str): Name for the set to be used in result DataFrame.
    - feature_name (str): The name of the feature column to aggregate.

    Returns:
    - pd.DataFrame: Result DataFrame containing aggregated counts and lists.

    Example:
    get_grouped_counts_feature_values(df, 'code_column', 'outfit_set', 'color')
    """
    result_df = df.groupby(code_sets).agg({
        code_sets: 'count',
        feature_name: list
    })

    result_df = result_df.rename(columns={code_sets: f'{name_set}_size'})
    result_df = result_df.rename(columns={feature_name: f'{feature_name}_list'})
    result_df.reset_index(inplace=True)
    result_df[f'{code_sets}_{feature_name}'] = list(zip(result_df[code_sets], result_df[f'{feature_name}_list']))
    result_df.drop(columns=[code_sets, f'{feature_name}_list'], inplace=True)

    final_df = result_df.groupby(f'{name_set}_size').agg({
        f'{name_set}_size': 'size',
        f'{code_sets}_{feature_name}': list
    })

    final_df = final_df.rename(columns={f'{name_set}_size': f'{name_set}_size_count'})
    final_df = final_df.rename(columns={f'{code_sets}_{feature_name}': f'{code_sets}_{feature_name}_tuple'})
    final_df = final_df.reset_index()

    return final_df


def get_unique_sets_features(list_set, feature_name):
    """
    Extracts unique sets of features from a list of sets and their respective
    counts.

    Parameters:
    - list_set (list): List of sets containing outfit codes and feature values.
    - feature_name (str): The name of the feature.

    Returns:
    - pd.DataFrame: DataFrame containing unique sets, outfit codes, and counts.

    Example:
    get_unique_sets_features([(1, ['red', 'blue']), (2, ['red', 'green']), ...], 'color')
    """
    unique_sets_features = {f'{feature_name}_set': [],
                            f'{feature_name}_outfit_codes': [],
                            f'{feature_name}_count': []}

    for value_set in list_set:
        if Counter(value_set[1]) not in unique_sets_features[f'{feature_name}_set']:
            unique_sets_features[f'{feature_name}_set'].append(Counter(value_set[1]))
            unique_sets_features[f'{feature_name}_outfit_codes'].append([value_set[0]])
            unique_sets_features[f'{feature_name}_count'].append(1)
        else:
            i = 0
            while unique_sets_features[f'{feature_name}_set'][i] != Counter(value_set[1]):
                i += 1
            unique_sets_features[f'{feature_name}_count'][i] += 1
            unique_sets_features[f'{feature_name}_outfit_codes'][i].append(value_set[0])

    unique_sets_features_df = pd.DataFrame(unique_sets_features)
    unique_sets_features_df = unique_sets_features_df.sort_values(by=f'{feature_name}_count', ascending=False)
    return unique_sets_features_df



def create_combinations(list_elements, root=True):
    """
    Generates all possible combinations of elements from a given list.

    Parameters:
    - list_elements (list): The list of elements for which combinations are generated.
    - root (bool): If True, includes combinations with a single element.

    Returns:
    - list: List of lists containing all possible combinations.

    Example:
    create_combinations(['A', 'B', 'C'])  # Output: [['A'], ['A', 'B'], ['A', 'C'], ['B'], ['B', 'C'], ['C']]
    """
    combination_list = [[list_elements[0]]]
    for element in list_elements[1:]:
        for combination in combination_list[:]:  # Make a copy to avoid modifying while iterating
            combination_list.append(combination + [element])
    if not root:
        for element in list_elements[1:]:
            combination_list.append([element])
    return combination_list



def create_configurations(configurations_base, optional_products):
    """
    Generates all possible configurations by combining base configurations
    with optional products.

    Parameters:
    - configurations_base (list): List of base configurations.
    - optional_products (list): List of optional products to be added to base configurations.

    Returns:
    - list: List of lists containing all possible configurations.

    Example:
    create_configurations(['Config1', 'Config2'], ['OptionalA', 'OptionalB'])
    # Output: [['Config1'], ['Config1', 'OptionalA'], ['Config1', 'OptionalB'],
    #          ['Config2'], ['Config2', 'OptionalA'], ['Config2', 'OptionalB']]
    """
    optional_configurations = create_combinations(optional_products, root=False)

    all_configurations = []

    for base in configurations_base:
        all_configurations.append(base)
        for optional in optional_configurations:
            all_configurations.append(base + optional)

    return all_configurations





