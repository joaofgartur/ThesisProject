import copy

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import logger, write_dataframe_to_csv, dict_to_dataframe
from .assessment import assess_all_surrogates, assess_model


class Pipeline:

    def __init__(self, dataset: Dataset, algorithm: Algorithm, model: object, settings: dict) -> None:
        self.dataset = dataset
        self.algorithm = algorithm
        self.settings = settings
        self.model = model
        self.results = None

    def __assessment__(self, train_set: Dataset,
                       validation_set: Dataset,
                       protected_feature: str,
                       algorithm: str = 'NA',
                       final_assessment: bool = False) -> pd.DataFrame:

        pipeline_details = {
            'dataset': train_set.name,
            'sensitive_attribute': protected_feature,
            'algorithm': algorithm
        }
        pipeline_df = dict_to_dataframe(pipeline_details)

        if final_assessment:
            decisions, assessment_df = assess_model(self.model, train_set, validation_set, protected_feature)
            print(f'Decisions: \n {decisions.value_counts()}')
        else:
            assessment_df = assess_all_surrogates(train_set, validation_set, protected_feature)

        return pd.concat([pipeline_df, assessment_df], axis=1)

    def __binary_attribute_mitigation__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset,
                                        protected_feature: str) -> pd.DataFrame:

        def _mitigate(_train_set: Dataset, _validation_set: Dataset, _protected_feature: str, _value: str) -> Dataset:

            _original_values = (_train_set.protected_attributes[_protected_feature],
                                _validation_set.protected_attributes[_protected_feature])

            _dummy_values = (_train_set.get_dummy_protected_feature(_protected_feature),
                             _validation_set.get_dummy_protected_feature(_protected_feature))

            _train_set.protected_attributes[_protected_feature] = _dummy_values[0][_value]
            _validation_set.protected_attributes[_protected_feature] = _dummy_values[1][_value]

            # define sensitive value
            _transformed_dataset = copy.deepcopy(_train_set)
            _transformed_dataset.set_feature(_protected_feature, _dummy_values[0][_value])
            self.algorithm.fit(_transformed_dataset, _protected_feature)
            _transformed_dataset = self.algorithm.transform(_transformed_dataset)

            _train_set.protected_attributes[_protected_feature] = _original_values[0]
            _transformed_dataset.protected_attributes[_protected_feature] = _original_values[0]
            _validation_set.protected_attributes[_protected_feature] = _original_values[1]

            return _transformed_dataset

        logger.info("[INTERVENTION] Correcting bias with binary algorithm.")

        results_df = pd.DataFrame()
        dummy_values = train_set.get_dummy_protected_feature(protected_feature)
        for value in dummy_values:
            logger.info(
                f"[INTERVENTION] Correcting bias w.r.t. attribute {protected_feature} for {value} with "
                f"{self.algorithm.__class__.__name__}")

            # bias mitigation
            transformed_dataset = _mitigate(train_set, validation_set, protected_feature, value)

            # surrogate models assessment
            logger.info("[POST-INTERVENTION] Assessing surrogate models.")
            assessment_results = self.__assessment__(transformed_dataset, validation_set, protected_feature,
                                                     self.algorithm.__class__.__name__)

            # final model assessment
            logger.info(f"[POST-INTERVENTION] Assessing model {self.model.__class__.__name__}.")
            final_model_results = self.__assessment__(transformed_dataset, test_set, protected_feature,
                                                      self.algorithm.__class__.__name__, True)

            value_results = pd.concat([assessment_results, final_model_results])
            results_df = pd.concat([results_df, value_results])

        return results_df

    def __multiclass_attribute_mitigation__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset,
                                            protected_feature: str) -> pd.DataFrame:
        logger.info("[INTERVENTION] Correcting bias with multi-value algorithm.")
        transformed_dataset = copy.deepcopy(train_set)

        # bias mitigation
        self.algorithm.set_validation_data(validation_set)
        self.algorithm.fit(transformed_dataset, protected_feature)
        transformed_dataset = self.algorithm.transform(transformed_dataset)

        # surrogate models assessment
        logger.info("[POST-INTERVENTION] Assessing surrogate models.")
        assessment_results = self.__assessment__(transformed_dataset, validation_set, protected_feature,
                                                 self.algorithm.__class__.__name__)

        # final model assessment
        logger.info(f"[POST-INTERVENTION] Assessing model {self.model.__class__.__name__}.")
        final_model_results = self.__assessment__(transformed_dataset, test_set, protected_feature,
                                                  self.algorithm.__class__.__name__, True)

        return pd.concat([assessment_results, final_model_results])

    def run(self) -> None:
        self.results = pd.DataFrame()

        try:
            logger.info("[PIPELINE] Start.")

            train_set, validation_set, test_set = self.dataset.split(self.settings)

            scaler = MinMaxScaler()
            scaler.fit(train_set.features)
            train_set = update_dataset(train_set, features=scaler.transform(train_set.features))
            validation_set = update_dataset(validation_set, features=scaler.transform(validation_set.features))
            test_set = update_dataset(test_set, features=scaler.transform(test_set.features))

            logger.info("[PRE-INTERVENTION] Assessing surrogate models.")
            for protected_feature in train_set.protected_attributes:
                self.results = pd.concat([self.results,
                                          self.__assessment__(train_set, validation_set, protected_feature)])

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

            logger.info("[PIPELINE] End.")

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
