import pandas as pd

from datasets import Dataset
from constants import POSITIVE_OUTCOME, PRIVILEGED, UNPRIVILEGED
from helpers import logger


class Compas(Dataset):
    _LOCAL_DATA_FILE = "datasets/local_storage/compas/compas-scores-two-years.csv"

    def __init__(self, dataset_info: dict):
        logger.info("Loading COMPAS dataset...")
        Dataset.__init__(self, dataset_info)
        self._load_dataset()
        self._prepare_dataset()
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

        data = self.__filter_data__(data)

        # select target and features
        self.targets = pd.DataFrame(data["two_year_recid"])
        self.features = data.drop(columns={"two_year_recid"})

    def _transform_dataset(self):

        def binarize_attribute(x, y):
            if x == y:
                return PRIVILEGED
            return UNPRIVILEGED

        def derive_class(x):
            return POSITIVE_OUTCOME - x

        for attribute, value in self.sensitive_attributes_info.items():
            self.features[attribute] = self.features[attribute].apply(lambda x, y=value: binarize_attribute(x, y))
        self.targets = self.targets.apply(lambda x: derive_class(x))

    def __filter_data__(self, data):
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
