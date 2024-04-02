"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset, update_dataset, match_features
from errors import error_check_dataset, error_check_sensitive_attribute
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import logger, bold, duplicate_rows, dict_to_dataframe
from evaluation import FairnessEvaluator


def assess_model(model: object, train_data: Dataset, validation_data: Dataset, sensitive_attribute: str):
    classifier_name = model.__class__.__name__

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model)
    ])

    validation_data = match_features(train_data, validation_data)

    x_train = train_data.features
    y_train = train_data.targets.to_numpy().ravel()

    if train_data.instance_weights is not None:
        pipeline.fit(x_train, y_train, classifier__sample_weight=train_data.instance_weights)
    else:
        pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(validation_data.features)
    predicted_data = update_dataset(validation_data, targets=predictions)

    fairness_evaluator = FairnessEvaluator(validation_data, predicted_data, sensitive_attribute)
    fairness_metrics = fairness_evaluator.evaluate()

    performance_evaluator = ModelEvaluator(validation_data, predicted_data)
    performance_metrics = performance_evaluator.evaluate()
    # performance_metrics = duplicate_rows(performance_metrics, fairness_metrics.shape[0])

    classifier_name = dict_to_dataframe({'classifier': classifier_name})
    # classifier_name = duplicate_rows(classifier_name, fairness_metrics.shape[0])

    results = pd.concat([classifier_name, fairness_metrics, performance_metrics], axis=1)

    return results, predictions


def assess_all_surrogates(train_set: Dataset,
                          validation_set: Dataset,
                          intervention_attribute: str = 'NA',
                          algorithm: str = 'NA'):
    """
    Conduct assessment on a dataset, including fairness evaluation and classifier accuracies.

    Parameters
    ----------
    validation_set
    algorithm
    intervention_attribute
    train_set : Dataset
        The dataset object containing features, targets, and sensitive attributes.

    Returns
    -------
    pd.DataFrame
        A dictionary containing fairness evaluation and classifier accuracies.

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
        - If there are missing sensitive attributes in the dataset.
        - If learning settings are missing required keys.
    """
    error_check_dataset(train_set)

    surrogate_classifiers = {
        #'LR': LogisticRegression(),
        #'SVC': SVC(),
        #'GNB': GaussianNB(),
        #'KNN': KNeighborsClassifier(),
        #"DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier()
    }

    global_results = pd.DataFrame()
    for feature in train_set.protected_features_names:

        for surrogate in surrogate_classifiers:
            logger.info(f'[ASSESSMENT] Assessing surrogate {surrogate} for feature \"{feature}\".')

            surrogate_model_results, _ = assess_model(
                surrogate_classifiers[surrogate],
                train_set,
                validation_set,
                feature)

            local_results = {
                'dataset': train_set.name,
                'sensitive_attribute': feature,
                'intervention_attribute': intervention_attribute,
                'algorithm': algorithm
            }
            local_results = dict_to_dataframe(local_results)
            # local_results = duplicate_rows(local_results, surrogate_model_results.shape[0])

            local_results = pd.concat([local_results, surrogate_model_results], axis=1)
            global_results = pd.concat([global_results, local_results])

    return global_results


def data_description_diff(df: pd.DataFrame, fixed_df: pd.DataFrame) -> pd.DataFrame:
    df_description = df.describe()
    fixed_df_description = fixed_df.describe()
    return df_description.compare(fixed_df_description)


def data_value_counts(df: pd.DataFrame) -> pd.Series:
    return df.value_counts()


def data_assessment(dataset: Dataset, fixed_dataset: Dataset, sensitive_attribute: str):

    print(f' --------- Assessment for {bold(sensitive_attribute)} --------- ')

    # compare features
    print(f'\t{bold("Features")}:')
    print(f'\t\t{bold("Data Description Differences")}: \n{data_description_diff(dataset.features, fixed_dataset.features).to_string()}')
    print(f'\t\t{bold("Data Value Counts")}: \n{data_value_counts(dataset.features)}')
    print(f'\t\t{bold("Fixed Data Value Counts")}: \n{data_value_counts(fixed_dataset.features)}')

    # compare targets
    print(f'\t{bold("Targets")}:')
    print(f'\t\t{bold("Data Description Differences")}: \n{data_description_diff(dataset.targets, fixed_dataset.targets).to_string()}')
    print(f'\t\t{bold("Data Value Counts")}: \n{data_value_counts(dataset.targets)}')
    print(f'\t\t{bold("Fixed Data Value Counts")}: \n{data_value_counts(fixed_dataset.targets)}')
