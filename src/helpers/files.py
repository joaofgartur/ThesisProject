import os
import pandas as pd
from datetime import datetime

from .random_numbers import get_seed


def extract_filename(filename):
    filename = filename.split(os.path.sep)[-1]
    return filename.split('.')[0].upper()


def create_directory(directory: str) -> None:
    mode = 0o777
    base_directory = ''
    for sub_directory in directory.split(os.path.sep):
        base_directory = os.path.join(base_directory, sub_directory)
        if not os.path.exists(base_directory):
            os.mkdir(base_directory, mode)


def write_dataframe_to_csv(df: pd.DataFrame, dataset_name: str, path: str) -> None:
    # use date and time to create path
    c = datetime.now()
    time = c.strftime('%d_%m_%y-%H_%M_%S')

    filename = f'seed_{get_seed()}_{dataset_name}_{time}.csv'

    if os.path.exists(path) & os.path.isdir(path):
        path = os.path.join(path, filename)
    else:
        create_directory(path)
        path = os.path.join(path, filename)

    df.to_csv(path, sep=',', index=False, encoding='utf-8')
