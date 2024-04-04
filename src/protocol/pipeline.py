import copy
import itertools

import pandas as pd

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import logger, write_dataframe_to_csv
from .assessment import assess_all_surrogates, assess_model, assess_dominance


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

            """
            scaler = MinMaxScaler(copy=False)
            scaler = scaler.fit(train_set.features)
            train_set.features = pd.DataFrame(scaler.transform(train_set.features),
                                              columns=train_set.features.columns)
            validation_set.features = pd.DataFrame(scaler.transform(validation_set.features),
                                                   columns=validation_set.features.columns)
            test_set.features = pd.DataFrame(scaler.transform(test_set.features),
                                             columns=test_set.features.columns)
            """

            logger.info("[PRE-INTERVENTION] Performing assessment...")

            self.results = assess_all_surrogates(train_set, validation_set)

            logger.info("[PRE-INTERVENTION] Assessment complete.")

            for protected_feature in train_set.protected_features_names:
                logger.info(f"[INTERVENTION] Correcting bias w.r.t. attribute {protected_feature} with "
                            f"{self.algorithm.__class__.__name__}")

                # preserve protected feature
                original_values = train_set.get_protected_feature(protected_feature)
                transformed_dataset = copy.deepcopy(train_set)
                logger.info(f"[INTERVENTION] Dominant is {0}")

                dummy_values = train_set.get_dummy_protected_feature(protected_feature)

                permutations = list(itertools.permutations(dummy_values.columns.to_list()))

                permutation_count = 1
                num_permutations = len(permutations)

                for permutation in permutations:
                    logger.info(f'[INTERVENTION] Permutation {permutation_count}/{num_permutations}')
                    temp_transformed_dataset = copy.deepcopy(train_set)

                    for value in permutation:
                        temp_transformed_dataset.set_feature(protected_feature, dummy_values[value])
                        self.algorithm.fit(temp_transformed_dataset, protected_feature)
                        temp_transformed_dataset = self.algorithm.transform(temp_transformed_dataset)

                    temp_transformed_dataset.set_feature(protected_feature, original_values)
                    if assess_dominance(transformed_dataset, temp_transformed_dataset, validation_set,
                                        protected_feature):
                        logger.info(f"[INTERVENTION] Dominant is {permutation_count}")
                        transformed_dataset = temp_transformed_dataset

                    permutation_count += 1

                logger.info('[INTERVENTION] Correction finished.')
                logger.info("[POST-INTERVENTION] Performing assessment...")

                surrogate_results = assess_all_surrogates(
                    transformed_dataset,
                    validation_set,
                    protected_feature,
                    self.algorithm.__class__.__name__)

                # data_assessment(train_set, fixed_dataset, feature)

                logger.info("[POST-INTERVENTION] Assessment complete.")

                final_model_results, decisions = assess_model(
                    self.model,
                    transformed_dataset,
                    test_set,
                    protected_feature
                )

                final_model_results = pd.DataFrame(final_model_results)

                self.results = pd.concat([self.results, surrogate_results, final_model_results])

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
