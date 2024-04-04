"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset, update_dataset, match_features
from errors import error_check_dataset
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import logger, bold, dict_to_dataframe, abs_diff
from evaluation import FairnessEvaluator

surrogate_models = {
    'LR': LogisticRegression(),
    'SVC': SVC(),
    'GNB': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier()
}


def compare_metric(series_a: pd.Series, series_b: pd) -> int:
    def compare(_a: float, _b: float) -> int:
        if _a > _b:
            return 1
        elif _a == _b:
            return 0
        return -1

    if len(series_a) == 1:
        return compare(series_a[0], series_b[0])

    a = abs_diff(series_a.min(), series_a.max())
    b = abs_diff(series_b.min(), series_b.max())

    return compare(a, b)


def assess_dominance(dominant: Dataset, candidate: Dataset, validation_set: Dataset,
                     sensitive_attribute: str) -> bool:
    count_dominant = 0
    count_candidate = 0

    for model in surrogate_models:

        dominant_fairness, dominant_performance = get_model_evaluators(surrogate_models[model],
                                                                       dominant,
                                                                       validation_set,
                                                                       sensitive_attribute)

        candidate_fairness, candidate_performance = get_model_evaluators(surrogate_models[model],
                                                                         candidate,
                                                                         validation_set,
                                                                         sensitive_attribute)

        dominant_f_metrics = dominant_fairness.evaluate(stats=False)
        candidate_f_metrics = candidate_fairness.evaluate(stats=False)

        for metric in dominant_f_metrics:
            if metric == 'Label':
                continue

            comparison_result = compare_metric(dominant_f_metrics[metric], candidate_f_metrics[metric])

            if comparison_result >= 0:
                count_candidate += 1
            else:
                count_dominant += 1

        # dominant_p_metrics = dominant_performance.evaluate().select_dtypes(include=['number'])
        # candidate_p_metrics = candidate_performance.evaluate().select_dtypes(include=['number'])

        """
        for metric in dominant_p_metrics:
            comparison_result = compare_metric(dominant_p_metrics[metric], candidate_p_metrics[metric])

            if comparison_result >= 0:
                count_dominant += 1
            else:
                count_candidate += 1
        """

    # print(f'dominant: {count_dominant} candidate: {count_candidate}')

    return True if count_candidate > count_dominant else False


def get_model_predictions(model: object, train_data: Dataset, validation_data: Dataset) -> Dataset:
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

    return predicted_data


def get_model_evaluators(model: object, train_data: Dataset, validation_data: Dataset, sensitive_attribute: str):
    predicted_data = get_model_predictions(model, train_data, validation_data)

    fairness_evaluator = FairnessEvaluator(validation_data, predicted_data, sensitive_attribute)

    performance_evaluator = ModelEvaluator(validation_data, predicted_data)

    return fairness_evaluator, performance_evaluator


def assess_model(model: object, train_data: Dataset, validation_data: Dataset, sensitive_attribute: str):
    model_name = model.__class__.__name__

    predicted_data = get_model_predictions(model, train_data, validation_data)

    fairness_evaluator, performance_evaluator = get_model_evaluators(model,
                                                                     train_data,
                                                                     validation_data,
                                                                     sensitive_attribute)
    fairness_metrics = fairness_evaluator.evaluate()

    performance_metrics = performance_evaluator.evaluate()

    model_name = dict_to_dataframe({'model': model_name})

    results = pd.concat([model_name, fairness_metrics, performance_metrics], axis=1)

    return results, predicted_data.targets


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

    global_results = pd.DataFrame()
    for feature in train_set.protected_features_names:

        for surrogate in surrogate_models:
            logger.info(f'[ASSESSMENT] Assessing surrogate {surrogate} for feature \"{feature}\".')

            surrogate_model_results, _ = assess_model(
                surrogate_models[surrogate],
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
    print(
        f'\t\t{bold("Data Description Differences")}: \n{data_description_diff(dataset.features, fixed_dataset.features).to_string()}')
    print(f'\t\t{bold("Data Value Counts")}: \n{data_value_counts(dataset.features)}')
    print(f'\t\t{bold("Fixed Data Value Counts")}: \n{data_value_counts(fixed_dataset.features)}')

    # compare targets
    print(f'\t{bold("Targets")}:')
    print(
        f'\t\t{bold("Data Description Differences")}: \n{data_description_diff(dataset.targets, fixed_dataset.targets).to_string()}')
    print(f'\t\t{bold("Data Value Counts")}: \n{data_value_counts(dataset.targets)}')
    print(f'\t\t{bold("Fixed Data Value Counts")}: \n{data_value_counts(fixed_dataset.targets)}')
