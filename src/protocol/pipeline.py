import copy

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import logger, write_dataframe_to_csv, dict_to_dataframe
from .assessment import assess_all_surrogates, assess_classifier


class Pipeline:

    def __init__(self, dataset: Dataset,
                 unbiasing_algorithms: [Algorithm],
                 surrogate_classifiers: dict,
                 test_classifier: object,
                 settings: dict) -> None:

        self.dataset = dataset
        self.unbiasing_algorithms = unbiasing_algorithms
        self.settings = settings
        self.test_classifier = test_classifier
        self.surrogate_classifiers = surrogate_classifiers
        self.results = {}

        self.unbiasing_algorithm: str = 'NA'
        self.set: str = 'NA'

    def __scale__(self, scaler: MinMaxScaler, train_data: Dataset, validation_data: Dataset, test_data: Dataset) \
            -> (Dataset, Dataset, Dataset):

        scaler.fit(train_data.features)

        train_data = update_dataset(train_data, features=scaler.transform(train_data.features))
        validation_data = update_dataset(validation_data, features=scaler.transform(validation_data.features))
        test_data = update_dataset(test_data, features=scaler.transform(test_data.features))

        return train_data, validation_data, test_data

    def __pre_intervention__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset) -> pd.DataFrame:
        logger.info("[PRE-INTERVENTION] Assessment.")

        protected_attributes = train_set.protected_attributes

        # surrogate_assessment
        df = pd.DataFrame()
        for attribute in protected_attributes:
            self.set = 'Validation'
            df = pd.concat([df, self.__assessment__(train_set, validation_set, attribute)])
            df = pd.concat([df, self.__assessment__(train_set, validation_set, attribute, True)])
            self.set = 'Test'
            df = pd.concat([df, self.__assessment__(train_set, test_set, attribute)])
            df = pd.concat([df, self.__assessment__(train_set, test_set, attribute, True)])

        return df

    def __post_intervention__(self, train_data: Dataset,
                              validation_data: Dataset,
                              test_data: Dataset,
                              protected_attribute: str) -> pd.DataFrame:
        logger.info("[POST-INTERVENTION] Assessment.")

        # surrogate models assessment
        self.set = 'Validation'
        val_assessment_results = self.__assessment__(train_data, validation_data, protected_attribute)
        val_final_model_results = self.__assessment__(train_data, validation_data, protected_attribute, True)

        self.set = 'Test'
        test_assessment_results = self.__assessment__(train_data, test_data, protected_attribute)
        test_final_model_results = self.__assessment__(train_data, test_data, protected_attribute, True)

        value_results = pd.concat([val_assessment_results,
                                   test_assessment_results,
                                   val_final_model_results,
                                   test_final_model_results])

        return value_results

    def __assessment__(self,
                       train_set: Dataset,
                       validation_set: Dataset,
                       protected_attribute: str,
                       test_assessment: bool = False) -> pd.DataFrame:

        # Surrogate Classifiers
        pipeline_details = {
            'dataset': self.dataset.name,
            'protected_attribute': protected_attribute,
            'unbiasing_algorithm': self.unbiasing_algorithm,
            'data': self.set,
            'classifier_type': 'Surrogate',
        }

        if test_assessment:
            decisions, test_classifier_df = assess_classifier(classifier=self.test_classifier,
                                                              train_data=train_set,
                                                              validation_data=validation_set,
                                                              protected_attribute=protected_attribute)
            print(f'Decisions: {decisions.value_counts()}')

            rows = len(test_classifier_df)
            pipeline_details['classifier_type'] = 'Test'
            pipeline_df = pd.concat([dict_to_dataframe(pipeline_details)] * rows, ignore_index=True)
            return pd.concat([pipeline_df, test_classifier_df], axis=1).reset_index(drop=True)

        else:
            surrogates_df = assess_all_surrogates(train_set=train_set,
                                                  validation_set=validation_set,
                                                  surrogate_classifiers=self.surrogate_classifiers,
                                                  protected_attribute=protected_attribute)
            rows = len(surrogates_df)
            pipeline_df = pd.concat([dict_to_dataframe(pipeline_details)] * rows, ignore_index=True)
            return pd.concat([pipeline_df, surrogates_df], axis=1).reset_index(drop=True)

    def __binary_attribute_mitigation__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset,
                                        unbiasing_algorithm: Algorithm, protected_attribute: str) -> pd.DataFrame:

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
            unbiasing_algorithm.fit(_transformed_dataset, _protected_feature)
            _transformed_dataset = unbiasing_algorithm.transform(_transformed_dataset)

            _train_set.protected_attributes[_protected_feature] = _original_values[0]
            _transformed_dataset.protected_attributes[_protected_feature] = _original_values[0]
            _validation_set.protected_attributes[_protected_feature] = _original_values[1]

            return _transformed_dataset

        logger.info("[INTERVENTION] Correcting bias with binary algorithm.")

        results_df = pd.DataFrame()
        dummy_values = train_set.get_dummy_protected_feature(protected_attribute)
        for value in dummy_values:
            logger.info(
                f"[INTERVENTION] Correcting bias w.r.t. attribute {protected_attribute} for {value} with "
                f"{unbiasing_algorithm.__class__.__name__}")

            # bias mitigation
            transformed_dataset = _mitigate(train_set, validation_set, protected_attribute, value)

            value_results = self.__post_intervention__(transformed_dataset, validation_set,
                                                       test_set, protected_attribute)

            results_df = pd.concat([results_df, value_results])

        return results_df

    def __multiclass_attribute_mitigation__(self, train_set: Dataset, validation_set: Dataset, test_set: Dataset,
                                            unbiasing_algorithm: Algorithm, protected_feature: str) -> pd.DataFrame:
        logger.info("[INTERVENTION] Correcting bias with multi-value algorithm.")
        transformed_dataset = copy.deepcopy(train_set)

        # bias mitigation
        unbiasing_algorithm.set_validation_data(validation_set)
        unbiasing_algorithm.fit(transformed_dataset, protected_feature)
        transformed_dataset = unbiasing_algorithm.transform(transformed_dataset)

        return self.__post_intervention__(transformed_dataset, validation_set, test_set, protected_feature)

    def run(self) -> None:
        self.results.update({unbiasing_algorithm.__class__.__name__: pd.DataFrame()
                             for unbiasing_algorithm in self.unbiasing_algorithms})

        try:
            logger.info("[PIPELINE] Start.")

            # split
            train_set, validation_set, test_set = self.dataset.split(self.settings)

            # scale
            train_set, validation_set, test_set = self.__scale__(MinMaxScaler(), train_set, validation_set, test_set)

            # pre-intervention
            pre_intervention_results = self.__pre_intervention__(train_set, validation_set, test_set)
            self.results.update(
                {unbiasing_algorithm.__class__.__name__: pre_intervention_results
                 for unbiasing_algorithm in self.unbiasing_algorithms})

            # correction
            for attribute in train_set.protected_features_names:

                for unbiasing_algorithm in self.unbiasing_algorithms:
                    self.unbiasing_algorithm = unbiasing_algorithm.__class__.__name__

                    if unbiasing_algorithm.is_binary:
                        attribute_results = self.__binary_attribute_mitigation__(train_set,
                                                                                 validation_set,
                                                                                 test_set,
                                                                                 unbiasing_algorithm,
                                                                                 attribute)
                    else:
                        attribute_results = self.__multiclass_attribute_mitigation__(train_set,
                                                                                     validation_set,
                                                                                     test_set,
                                                                                     unbiasing_algorithm,
                                                                                     attribute)

                    self.results[unbiasing_algorithm.__class__.__name__] = pd.concat(
                        [self.results[unbiasing_algorithm.__class__.__name__], attribute_results])

                    self.save_algorithm_results(unbiasing_algorithm.__class__.__name__, attribute=attribute)

            logger.info("[PIPELINE] End.")

        except Exception as e:
            logger.error(f'An error occurred in the pipeline: \n {e}')
            raise

    def save(self, path: str = 'results') -> None:
        if self.results is not None:
            for key, result in self.results.items():
                write_dataframe_to_csv(df=result, dataset_name=self.dataset.name,
                                       path=f'{path}/{key}')
        else:
            raise KeyError('Results are empty!')

    def save_algorithm_results(self, algorithm: str, attribute: str, path: str = 'results') -> None:
        if self.results is not None:
            suffix = f'_intermediary_{attribute}_'
            write_dataframe_to_csv(df=self.results[algorithm],
                                   dataset_name=self.dataset.name+suffix,
                                   path=f'{path}/{algorithm}')
        else:
            raise KeyError('Results are empty!')

    def run_and_save(self, path: str = 'results') -> None:
        self.run()
        self.save(path)
