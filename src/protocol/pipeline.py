import copy
import itertools

import pandas as pd

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import logger, write_dataframe_to_csv, dict_to_dataframe
from .assessment import assess_all_surrogates, assess_model


class Pipeline:

    def __init__(self, dataset: Dataset, algorithm: Algorithm, model: object, settings: dict) -> None:
        self.dataset = dataset
        self.algorithm = algorithm
        self.settings = settings
        self.model = model
        self.results = None
        self.TRAIN = 0
        self.VALIDATION = 1
        self.TEST = 2
        self.protected_feature = []

    def __assessment__(self, train_set: Dataset, validation_set: Dataset,
                       algorithm: str = 'NA'):
        results = pd.DataFrame()
        protected_feature = self.protected_feature[0]

        train_dummy_values = train_set.get_dummy_protected_feature(protected_feature)
        validation_dummy_values = validation_set.get_dummy_protected_feature(protected_feature)

        train_original_values = train_set.protected_features[protected_feature]
        validation_original_values = validation_set.protected_features[protected_feature]

        for value in train_dummy_values:
            train_set.protected_features[protected_feature] = train_dummy_values[value]
            validation_set.protected_features[protected_feature] = validation_dummy_values[value]

            value_results = {
                'dataset': train_set.name,
                'sensitive_attribute': protected_feature,
                'value': value,
                'algorithm': algorithm
            }

            value_results = dict_to_dataframe(value_results)

            assessment_results = assess_all_surrogates(
                train_set,
                validation_set,
                protected_feature)
            assessment_results = pd.concat([value_results, assessment_results], axis=1)

            results = pd.concat([results, assessment_results])

        train_set.protected_features[protected_feature] = train_original_values
        validation_set.protected_features[protected_feature] = validation_original_values

        return results

    def __binary_attribute_mitigation__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset,
                                        protected_feature: str) -> pd.DataFrame:

        def _mitigate(_train_set: Dataset, _validation_set: Dataset) -> Dataset:
            _protected_feature, _value = self.protected_feature

            original_values = (_train_set.protected_features[_protected_feature],
                               _validation_set.protected_features[_protected_feature])

            dummy_values = (_train_set.get_dummy_protected_feature(_protected_feature),
                            _validation_set.get_dummy_protected_feature(_protected_feature))

            _train_set.protected_features[_protected_feature] = dummy_values[self.TRAIN][_value]
            _validation_set.protected_features[_protected_feature] = dummy_values[self.VALIDATION][_value]

            # define sensitive value
            _transformed_dataset = copy.deepcopy(_train_set)
            _transformed_dataset.set_feature(_protected_feature, dummy_values[self.TRAIN][_value])
            self.algorithm.fit(_transformed_dataset, _protected_feature)
            _transformed_dataset = self.algorithm.transform(_transformed_dataset)

            _train_set.protected_features[_protected_feature] = original_values[self.TRAIN]
            _transformed_dataset.protected_features[_protected_feature] = original_values[self.TRAIN]
            _validation_set.protected_features[_protected_feature] = original_values[self.VALIDATION]

            return _transformed_dataset

        results = pd.DataFrame()

        test_dummy_values = test_set.get_dummy_protected_feature(protected_feature)
        test_original_values = test_set.protected_features[protected_feature]

        for value in test_dummy_values:
            logger.info(
                f"[INTERVENTION] Correcting bias w.r.t. attribute {protected_feature} for {value} with "
                f"{self.algorithm.__class__.__name__}")

            self.protected_feature = [protected_feature, value]

            transformed_dataset = _mitigate(train_set, validation_set)

            assessment_results = self.__assessment__(transformed_dataset, validation_set,
                                                     self.algorithm.__class__.__name__)

            # prepare test set
            test_set.protected_features[protected_feature] = test_dummy_values[value]
            final_model_results = self.__train_final_model__(transformed_dataset, test_set)
            test_set.protected_features[protected_feature] = test_original_values

            value_results = pd.concat([assessment_results, final_model_results])
            results = pd.concat([results, value_results])

        return results

    def __multiclass_attribute_mitigation__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset,
                                            attribute: str) -> pd.DataFrame:
        """"""
        pass

    def __train_final_model__(self, train_set: Dataset, test_set: Dataset) -> pd.DataFrame:
        protected_feature = self.protected_feature[0]

        logger.info("[POST-INTERVENTION] Performing assessment...")

        final_model_results, decisions = assess_model(
            self.model,
            train_set,
            test_set,
            protected_feature
        )

        final_model_results = pd.DataFrame(final_model_results)

        logger.info("[POST-INTERVENTION] Assessment complete.")

        print(f'Decisions: {decisions}')

        return final_model_results

    def run(self) -> None:

        try:
            train_set, validation_set, test_set = self.dataset.split(self.settings)

            logger.info("[PRE-INTERVENTION] Performing assessment...")

            self.results = pd.DataFrame()
            for protected_feature in train_set.protected_features:
                self.protected_feature = [protected_feature, '']
                self.results = pd.concat(
                    [self.results, self.__assessment__(train_set, validation_set, protected_feature)])

            for protected_feature in train_set.protected_features_names:

                if self.algorithm.is_binary:
                    attribute_results = self.__binary_attribute_mitigation__(train_set,
                                                                             validation_set,
                                                                             test_set,
                                                                             protected_feature)
                else:
                    attribute_results = self.__multiclass_attribute_mitigation__(train_set,
                                                                                 validation_set,
                                                                                 test_set,
                                                                                 protected_feature)

                self.results = pd.concat([self.results, attribute_results])

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
