import pandas as pd
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from datasets import Dataset
from helpers import logger, extract_filename


class AIF360AdultIncome(Dataset):

    def _load_dataset(self):
        data = load_preproc_data_adult()

        features = pd.DataFrame(data.features, columns=data.feature_names)
        targets = pd.DataFrame(data.labels, columns=data.label_names)

        return features, targets

    def _transform_protected_attributes(self):
        pass

    def __init__(self, dataset_info: dict):
        logger.info(f'[{extract_filename(__file__)}] Loading...')
        Dataset.__init__(self, dataset_info)