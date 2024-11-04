"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import pandas as pd

from utils import logger


def bold(string: str) -> str:
    bold_start = "\033[1m"
    bold_end = "\033[0m"

    return bold_start + string + bold_end


def extract_value(key: str, dictionary: dict):
    if key in dictionary.keys():
        return dictionary[key]
    logger.error(f'Missing information: {key}')
    raise ValueError


def duplicate_rows(df: pd.DataFrame, num_duplicates: int) -> pd.DataFrame:
    return pd.concat([df] * num_duplicates, ignore_index=True)


def dict_to_dataframe(dictionary: dict) -> pd.DataFrame:
    return pd.DataFrame(dictionary, index=[0])


def concat_df(df1: pd.DataFrame, df2: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    return pd.concat([df1, df2], axis=axis).reset_index(drop=True)


def round_df(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    df = df.round(decimals)
    padding_columns = df.select_dtypes(include=['float']).columns
    df[padding_columns] = df[padding_columns].applymap(lambda x: f"{x:.4f}")
    return df
