import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import logger, write_dataframe_to_csv
from .assessment import assess_all_surrogates, data_assessment
from .model_testing import get_model_decisions


class Pipeline:

    def __init__(self, dataset: Dataset, algorithm: Algorithm, model: object, settings: dict) -> None:
        self.dataset = dataset
        self.algorithm = algorithm
        self.settings = settings
        self.model = model
        self.results = None

    def run(self) -> None:

        try:
            train_set, validation_set, test_set = self.dataset.split(self.settings)

            scaler = MinMaxScaler(copy=False)
            scaler = scaler.fit(train_set.features)
            train_set.features = pd.DataFrame(scaler.transform(train_set.features),
                                              columns=train_set.features.columns)
            validation_set.features = pd.DataFrame(scaler.transform(validation_set.features),
                                                   columns=validation_set.features.columns)
            test_set.features = pd.DataFrame(scaler.transform(test_set.features),
                                             columns=test_set.features.columns)

            logger.info("[PRE-INTERVENTION] Performing assessment...")

            self.results = assess_all_surrogates(train_set, validation_set)

            logger.info("[PRE-INTERVENTION] Assessment complete.")

            for feature in train_set.protected_features:
                logger.info(f"[INTERVENTION] Correcting bias w.r.t. attribute {feature} with "
                            f"{self.algorithm.__class__.__name__}")

                fixed_dataset = self.algorithm.repair(train_set, feature)

                logger.info('[INTERVENTION] Correction finished.')
                logger.info("[POST-INTERVENTION] Performing assessment...")

                results = assess_all_surrogates(
                    fixed_dataset,
                    validation_set,
                    feature,
                    self.algorithm.__class__.__name__)
                self.results = pd.concat([self.results, results])
                data_assessment(train_set, fixed_dataset, feature)

                logger.info("[POST-INTERVENTION] Assessment complete.")

                decisions = get_model_decisions(self.model, fixed_dataset, test_set)
                print(decisions)

        except Exception as e:
            logger.error(f'An error occurred in the pipeline: \n {e}')
            raise

    def save(self, path: str = 'results') -> None:
        if self.results is not None:
            write_dataframe_to_csv(df=self.results, dataset_name=self.dataset.name,
                                   path=path)
        else:
            raise KeyError('Results are empty!')

    def run_and_save(self, path: str = 'results') -> None:
        self.run()
        self.save(path)
