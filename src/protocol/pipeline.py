import copy
import itertools
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import logger, write_dataframe_to_csv, dict_to_dataframe
from .assessment import classifier_assessment


class Pipeline:

    def __init__(self, dataset: Dataset,
                 unbiasing_algorithms: [Algorithm],
                 surrogate_classifiers: list[object],
                 test_classifier: object) -> None:

        self.dataset = dataset
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        self.unbiasing_algorithms = unbiasing_algorithms

        self.surrogate_classifiers = surrogate_classifiers
        self.test_classifier = test_classifier

        self.results = {}

        self.unbiasing_algorithm: str = 'NA'
        self.set: str = 'NA'

    def __scale__(self, scaler: MinMaxScaler):

        scaler.fit(self.train_set.features)

        self.train_data = update_dataset(self.train_set, features=scaler.transform(self.train_set.features))
        self.validation_data = update_dataset(self.validation_set,
                                              features=scaler.transform(self.validation_set.features))
        self.test_data = update_dataset(self.test_set, features=scaler.transform(self.test_set.features))

    def __pre_intervention__(self, train_set: Dataset) -> pd.DataFrame:
        logger.info("[PRE-INTERVENTION] Assessment.")

        df = pd.DataFrame()
        for attribute in train_set.protected_attributes:
            self.set = 'Validation'
            df = pd.concat([df, self.__assessment__(train_set, self.validation_set, attribute)])
            df = pd.concat([df, self.__assessment__(train_set, self.validation_set, attribute, True)])
            self.set = 'Test'
            df = pd.concat([df, self.__assessment__(train_set, self.test_set, attribute)])
            df = pd.concat([df, self.__assessment__(train_set, self.test_set, attribute, True)])

        return df

    def __post_intervention__(self, train_data: Dataset,
                              protected_attribute: str) -> pd.DataFrame:
        logger.info("[POST-INTERVENTION] Assessment.")

        # surrogate models assessment
        self.set = 'Validation'
        val_assessment_results = self.__assessment__(train_data, self.validation_set, protected_attribute)
        val_final_model_results = self.__assessment__(train_data, self.validation_set, protected_attribute, True)

        self.set = 'Test'
        test_assessment_results = self.__assessment__(train_data, self.test_set, protected_attribute)
        test_final_model_results = self.__assessment__(train_data, self.test_set, protected_attribute, True)

        return pd.concat([val_assessment_results, test_assessment_results, val_final_model_results,
                          test_final_model_results])

    def __assessment__(self,
                       train_set: Dataset,
                       validation_set: Dataset,
                       protected_attribute: str,
                       test_assessment: bool = False) -> pd.DataFrame:

        def classifier_assessment_wrapper(args):
            _classifier, _train_data, _validation_data, _protected_attribute = args
            return classifier_assessment(_classifier, _train_data, _validation_data, protected_attribute)[1]

        # Surrogate Classifiers
        pipeline_details = {
            'dataset': self.dataset.name,
            'protected_attribute': protected_attribute,
            'unbiasing_algorithm': self.unbiasing_algorithm,
            'data': self.set,
            'classifier_type': 'Surrogate' if test_assessment else 'Test',
        }

        classifiers = [self.test_classifier] if test_assessment else self.surrogate_classifiers

        num_threads = len(classifiers)
        args_list = [(classifier, train_set, validation_set, protected_attribute) for classifier in classifiers]
        results = pd.DataFrame()
        with (ThreadPoolExecutor(max_workers=num_threads) as executor):
            results = pd.concat([results, *list(executor.map(classifier_assessment_wrapper, args_list))]
                                ).reset_index(drop=True)

        pipeline_df = pd.concat([dict_to_dataframe(pipeline_details)] * len(results), ignore_index=True)

        return pd.concat([pipeline_df, results], axis=1).reset_index(drop=True)

    def __bias_reduction__(self, train_set: Dataset, validation_set: Dataset,
                           unbiasing_algorithm: Algorithm, protected_attribute: str) -> pd.DataFrame:

        unbiasing_algorithm_type = 'binary' if unbiasing_algorithm.is_binary else 'multi-class'
        logger.info(f'[INTERVENTION] Reducing bias for {unbiasing_algorithm_type} '
                    f'sensitive attribute {protected_attribute} with unbiasing algorithm'
                    f' {unbiasing_algorithm.__class__.__name__}.')

        metrics_df = pd.DataFrame()
        if unbiasing_algorithm.is_binary:
            attribute_values = train_set.get_dummy_protected_feature(protected_attribute)
        else:
            attribute_values = pd.DataFrame(train_set.get_protected_feature(protected_attribute),
                                            columns=[protected_attribute])

        for value in attribute_values:
            if unbiasing_algorithm.is_binary:
                logger.info(
                    f"[INTERVENTION] Correcting bias w.r.t. attribute {protected_attribute} for {value} with "
                    f"{unbiasing_algorithm.__class__.__name__}")

            transformed_dataset = copy.deepcopy(train_set)
            transformed_dataset.set_feature(protected_attribute, attribute_values[value])

            if unbiasing_algorithm.needs_auxiliary_data:
                unbiasing_algorithm.auxiliary_data = validation_set

            unbiasing_algorithm.fit(transformed_dataset, protected_attribute)
            transformed_dataset = unbiasing_algorithm.transform(transformed_dataset)

            if transformed_dataset.error_flag:
                logger.error(f'[INTERVENTION] Error occurred bias correction w.r.t. attribute {protected_attribute} '
                             f'for {value} with "{unbiasing_algorithm.__class__.__name__}"')
                continue

            value_results = self.__post_intervention__(transformed_dataset, protected_attribute)

            metrics_df = pd.concat([metrics_df, value_results])

        return metrics_df

    def run(self) -> None:
        self.results.update({unbiasing_algorithm.__class__.__name__: pd.DataFrame()
                             for unbiasing_algorithm in self.unbiasing_algorithms})

        try:
            logger.info("[PIPELINE] Start.")

            # split
            self.train_set, self.validation_set, self.test_set = self.dataset.split()

            # scale
            self.__scale__(MinMaxScaler())

            # pre-intervention
            pre_intervention_results = self.__pre_intervention__(self.train_set)
            self.results.update(
                {unbiasing_algorithm.__class__.__name__: pre_intervention_results
                 for unbiasing_algorithm in self.unbiasing_algorithms})

            # correction
            for attribute, unbiasing_algorithm in list(itertools.product(self.train_set.protected_features_names,
                                                                         self.unbiasing_algorithms)):
                self.unbiasing_algorithm = unbiasing_algorithm.__class__.__name__

                attribute_results = self.__bias_reduction__(self.train_set,
                                                            self.validation_set,
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
                                   dataset_name=self.dataset.name + suffix,
                                   path=f'{path}/{algorithm}')
        else:
            raise KeyError('Results are empty!')

    def run_and_save(self, path: str = 'results') -> None:
        self.run()
        self.save(path)
