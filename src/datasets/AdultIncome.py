import pandas as pd
from ucimlrepo import fetch_ucirepo

from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset


class AdultIncome(Dataset):
    _LOCAL_DATA_FILE = "datasets/local_storage/adult_income/adult.data"

    def __init__(self, dataset_info: dict):
        Dataset.__init__(self, dataset_info)
        self._load_dataset()
        self._prepare_dataset()

    def _load_dataset(self):
        try:
            dataset = fetch_ucirepo(id=2)
            self.features = dataset.data.features
            self.targets = dataset.data.targets
        except ConnectionError:
            dataset = pd.read_csv(self._LOCAL_DATA_FILE, header=None)
            labels = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex",
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            dataset = dataset.set_axis(labels=labels, axis="columns")

            self.targets = pd.DataFrame(dataset["income"])
            self.features = dataset.drop(columns={"income"})

    def _transform_dataset(self):
        self.__derive_classes__()

    def __derive_classes__(self):
        self.targets = (self.targets
                        .replace("<=", NEGATIVE_OUTCOME, regex=True)
                        .replace(">", POSITIVE_OUTCOME, regex=True)
                        .astype('int64'))
