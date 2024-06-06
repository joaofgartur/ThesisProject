import pandas as pd
from ucimlrepo import fetch_ucirepo

from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset
from helpers import logger, extract_filename


class AdultIncome(Dataset):
    _LOCAL_DATA_FILE = "datasets/local_storage/adult_income/adult.data"

    def __init__(self, dataset_info: dict):
        logger.info(f'[{extract_filename(__file__)}] Loading...')
        Dataset.__init__(self, dataset_info)

    def _load_dataset(self):
        try:
            dataset = fetch_ucirepo(id=2)
            return dataset.data.features, dataset.data.targets
        except ConnectionError:
            logger.error(f'[{extract_filename(__file__)}] Failed to load from online source.'
                         f' Loading from local storage.')

            dataset = pd.read_csv(self._LOCAL_DATA_FILE, header=None)
            labels = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex",
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            dataset = dataset.set_axis(labels=labels, axis="columns")

            targets = pd.DataFrame(dataset["income"])
            features = dataset.drop(columns=["income"])

            return features, targets

    def _transform_protected_attributes(self):

        self.targets = (self.targets.replace("<=", NEGATIVE_OUTCOME, regex=True)
                        .replace(">", POSITIVE_OUTCOME, regex=True)
                        .astype('int64'))
