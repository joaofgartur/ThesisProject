import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from ucimlrepo import fetch_ucirepo

from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME, PRIVILEGED, UNPRIVILEGED
from datasets import Dataset, is_privileged
from helpers import logger


class AdultIncome(Dataset):
    _LOCAL_DATA_FILE = "datasets/local_storage/adult_income/adult.data"

    def __init__(self, dataset_info: dict):
        logger.info("Loading Adult Income dataset...")
        Dataset.__init__(self, dataset_info)
        logger.info("Dataset loaded.")

    def _load_dataset(self):
        try:
            dataset = fetch_ucirepo(id=2)
            return dataset.data.features, dataset.data.targets
        except ConnectionError:
            logger.error("Failed to load from online source!")
            logger.info("Loading dataset from local storage...")

            dataset = pd.read_csv(self._LOCAL_DATA_FILE, header=None)
            labels = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex",
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            dataset = dataset.set_axis(labels=labels, axis="columns")

            targets = pd.DataFrame(dataset["income"])
            features = dataset.drop(columns=["income"])

            return features, targets

    def _pre_process_dataset(self):
        """
        Pre-processing method based on 'Three naive Bayes approaches for discrimination-free
        classification'
        Returns
        -------

        """
        numerical_data = self.features.select_dtypes(include='number')
        categorical_data = self.features.drop(numerical_data, axis=1)

        # discretize numerical attributes
        discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
        numerical_data_discretized = pd.DataFrame(discretizer.fit_transform(numerical_data),
                                                  columns=numerical_data.columns)

        # Combine discretized numerical attributes with categorical attributes
        data_preprocessed = pd.concat([categorical_data, numerical_data_discretized], axis=1)

        # Handling low frequency counts (pooling)
        for column in data_preprocessed.columns:
            counts = data_preprocessed[column].value_counts()
            infrequent_values = counts[counts < 50].index
            data_preprocessed.loc[data_preprocessed[column].isin(infrequent_values), column] = 'pool'

        self.features = data_preprocessed

    def _transform_protected_attributes(self):

        # binarize attribute
        for feature, value in zip(self.protected_features, self.privileged_classes):
            self.features[feature] = self.features[feature].apply(lambda x, y=value: is_privileged(x, y))

        self.targets = (self.targets.replace("<=", NEGATIVE_OUTCOME, regex=True)
                        .replace(">", POSITIVE_OUTCOME, regex=True)
                        .astype('int64'))
