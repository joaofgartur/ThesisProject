import os
import shutil

import pandas as pd

from .random_numbers import get_seed


def backup_dataset(dataset, path):
    seed = get_seed()
    write_dataframe_to_csv(dataset.features, f'{seed}_features', path)
    write_dataframe_to_csv(dataset.targets, f'{seed}_targets', path)
    write_dataframe_to_csv(dataset.protected_features, f'{seed}_protected_features', path)


def restore_dataset(dataset, path):
    seed = get_seed()
    dataset.features = read_csv_to_dataframe(f'{seed}_features', path)
    dataset.targets = read_csv_to_dataframe(f'{seed}_targets', path)
    dataset.protected_features = read_csv_to_dataframe(f'{seed}_protected_features', path)


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


def delete_directory(path: str) -> None:
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def write_dataframe_to_csv(df: pd.DataFrame, filename: str, path: str) -> None:
    if os.path.exists(path) & os.path.isdir(path):
        path = os.path.join(path, filename)
    else:
        create_directory(path)
        path = os.path.join(path, filename)

    file_exists = os.path.isfile(path)
    with open(path, 'a', buffering=1) as f:
        df.to_csv(f, sep=',', index=False, encoding='utf-8', mode='a', header=not file_exists)
        f.flush()


def read_csv_to_dataframe(filename: str, path: str) -> pd.DataFrame:
    path = os.path.join(path, filename)
    return pd.read_csv(path)
