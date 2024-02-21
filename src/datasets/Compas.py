import pandas as pd

from datasets import Dataset, is_privileged
from constants import POSITIVE_OUTCOME
from helpers import logger


def filter_data(data):
    """
    Method that filter the dataset according to the approach taken by ProPublica.
    :return:
    """
    selected_columns = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
                        'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in',
                        'c_jail_out']
    filtered_data = data[selected_columns]

    filtered_data = filtered_data[(filtered_data['days_b_screening_arrest'] <= 30) &
                                  (filtered_data['days_b_screening_arrest'] >= -30) &
                                  (filtered_data['is_recid'] != -1) &
                                  (filtered_data['c_charge_degree'] != "O") &
                                  (filtered_data['score_text'] != 'N/A')]

    return filtered_data


class Compas(Dataset):

    _LOCAL_DATA_FILE = "datasets/local_storage/compas/compas-scores-two-years.csv"

    def __init__(self, dataset_info: dict):
        logger.info("Loading COMPAS dataset...")
        Dataset.__init__(self, dataset_info)
        logger.info("Dataset loaded.")

    def _load_dataset(self):
        try:
            dataset_url = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
                           "master/compas-scores-two-years.csv")
            data = pd.read_csv(dataset_url)
        except Exception:
            logger.error("Failed to load dataset from online source!")
            logger.info("Loading dataset from local storage...")

            data = pd.read_csv(self._LOCAL_DATA_FILE)

        # filter data
        data = filter_data(data)

        # select target and features
        targets = pd.DataFrame(data[self.target])
        features = data.drop(columns={self.target})

        return features, targets

    def _pre_process_dataset(self):
        pass

    def _transform_protected_attributes(self):

        def derive_class(x):
            return POSITIVE_OUTCOME - x

        # binarize attribute
        for feature, value in zip(self.protected_features, self.privileged_classes):
            self.features[feature] = self.features[feature].apply(lambda x, y=value: is_privileged(x, y))

        self.targets = self.targets.apply(lambda x: derive_class(x))
