from collections import Counter
from itertools import combinations

import pandas as pd


def get_grouped_counts(df: pd.DataFrame, code_sets: str, name_set: str) -> pd.DataFrame:
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
    set_size = (
        df.groupby(code_sets)[code_sets].count().reset_index(name=f"{name_set}_size")
    )
    grouped_set_size = (
        set_size.groupby(f"{name_set}_size")
        .size()
        .reset_index(name=f"grouped_{name_set}_size")
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
    result_df = df.groupby(code_sets).agg({code_sets: "count", feature_name: list})

    result_df = result_df.rename(columns={code_sets: f"{name_set}_size"})
    result_df = result_df.rename(columns={feature_name: f"{feature_name}_list"})
    result_df.reset_index(inplace=True)
    result_df[f"{code_sets}_{feature_name}"] = list(
        zip(result_df[code_sets], result_df[f"{feature_name}_list"])
    )
    result_df.drop(columns=[code_sets, f"{feature_name}_list"], inplace=True)

    final_df = result_df.groupby(f"{name_set}_size").agg(
        {f"{name_set}_size": "size", f"{code_sets}_{feature_name}": list}
    )

    final_df = final_df.rename(columns={f"{name_set}_size": f"{name_set}_size_count"})
    final_df = final_df.rename(
        columns={f"{code_sets}_{feature_name}": f"{code_sets}_{feature_name}_tuple"}
    )
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
    unique_sets_features = {
        f"{feature_name}_set": [],
        f"{feature_name}_outfit_codes": [],
        f"{feature_name}_count": [],
    }

    for value_set in list_set:
        if Counter(value_set[1]) not in unique_sets_features[f"{feature_name}_set"]:
            unique_sets_features[f"{feature_name}_set"].append(Counter(value_set[1]))
            unique_sets_features[f"{feature_name}_outfit_codes"].append([value_set[0]])
            unique_sets_features[f"{feature_name}_count"].append(1)
        else:
            i = 0
            while unique_sets_features[f"{feature_name}_set"][i] != Counter(
                value_set[1]
            ):
                i += 1
            unique_sets_features[f"{feature_name}_count"][i] += 1
            unique_sets_features[f"{feature_name}_outfit_codes"][i].append(value_set[0])

    unique_sets_features_df = pd.DataFrame(unique_sets_features)
    unique_sets_features_df = unique_sets_features_df.sort_values(
        by=f"{feature_name}_count", ascending=False
    )
    return unique_sets_features_df


def create_combinations(elements):
    """All nonempty subsets of `elements`, smallest first."""
    elements = list(elements)
    return [
        list(subset)
        for size in range(1, len(elements) + 1)
        for subset in combinations(elements, size)
    ]


def create_configurations(configurations_base, optional_products):
    """Combine each base with every subset of optional product classes."""
    optional = [[], *create_combinations(optional_products)]
    return [list(base) + extra for base in configurations_base for extra in optional]


def select_valid_outfits(df_outfit_products, feature_name, code_sets, all_configurations):
    """Keep outfits whose multiset of classes equals one of the configurations.

    Returns the selected rows and the number of outfits per configuration index.
    """
    signatures = {}
    for index, configuration in enumerate(all_configurations):
        signatures.setdefault(frozenset(Counter(configuration).items()), index)
    selected, counts = [], {}
    for outfit_id, rows in df_outfit_products.groupby(code_sets):
        signature = frozenset(Counter(rows[feature_name]).items())
        if signature in signatures:
            index = signatures[signature]
            selected.append(outfit_id)
            counts[index] = counts.get(index, 0) + 1
    columns = [code_sets, "cod_modelo_color", feature_name, "des_filename"]
    return (
        df_outfit_products.loc[
            df_outfit_products[code_sets].isin(selected), columns
        ].copy(),
        counts,
    )
