import pandas as pd

from helpers import logger


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
