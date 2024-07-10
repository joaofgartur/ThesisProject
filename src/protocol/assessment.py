"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datasets import Dataset, update_dataset, match_features
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import bold, dict_to_dataframe, concat_df
from evaluation import FairnessEvaluator


def get_classifier_predictions(model: object, train_data: Dataset, validation_data: Dataset) -> Dataset:

    pipeline = Pipeline([
        ('classifier', model)
    ], memory=None)

    validation_data = match_features(train_data, validation_data)

    x_train = train_data.features.to_numpy().astype(np.float32)
    y_train = train_data.targets.to_numpy().astype(np.float32).ravel()

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(validation_data.features.to_numpy())
    predicted_data = update_dataset(validation_data, targets=predictions)

    return predicted_data


def fairness_assessment(data: Dataset, predictions: Dataset, sensitive_attribute: str) -> pd.DataFrame:
    original_attribute_values = data.protected_features[sensitive_attribute]

    dummy_values = data.get_dummy_protected_feature(sensitive_attribute)

    assessment_df = pd.DataFrame()
    for value in dummy_values:
        data.protected_features[sensitive_attribute] = dummy_values[value]
        fairness_evaluator = FairnessEvaluator(data, predictions, sensitive_attribute)
        value_df = pd.concat([dict_to_dataframe({'value': value}), fairness_evaluator.evaluate()], axis=1)
        assessment_df = pd.concat([assessment_df, value_df])

    data.protected_features[sensitive_attribute] = original_attribute_values

    return assessment_df.reset_index(drop=True)


def distribution_assessment(train_data: Dataset, predicted_data: Dataset, predictions_data: Dataset, protected_attribute: str = 'NA') -> pd.DataFrame:
    def get_value_counts(df: pd.DataFrame) -> pd.DataFrame:
        ratios = df.value_counts(normalize=True).reset_index()
        attribute, target = ratios.columns[0], ratios.columns[1]
        ratios['column_name'] = ratios[attribute] + '_' + ratios[target].astype(str)
        ratios.drop(columns=[attribute, target], inplace=True)
        ratios = ratios.set_index('column_name').T.reset_index(drop=True)
        ratios = ratios.reindex(sorted(ratios.columns), axis=1)
        return ratios

    def map_protected_attribute(value):
        return train_data.features_mapping[protected_attribute][value]

    train_pg = pd.concat([train_data.protected_features[protected_attribute].apply(map_protected_attribute),
                          train_data.targets], axis=1)
    train_pg_ratios = get_value_counts(train_pg)
    train_pg_ratios = train_pg_ratios.add_suffix('_a_train_set_')

    predicted_pg = pd.concat([predicted_data.protected_features[protected_attribute].apply(map_protected_attribute),
                              predicted_data.targets], axis=1)
    predicted_pg_ratios = get_value_counts(predicted_pg)
    predicted_pg_ratios = predicted_pg_ratios.add_suffix('_b_predicted_set_')

    predictions_pg = pd.concat(
        [predictions_data.protected_features[protected_attribute].apply(map_protected_attribute),
         predictions_data.targets * 1.0], axis=1)
    predictions_pg_ratios = get_value_counts(predictions_pg)
    predictions_pg_ratios = predictions_pg_ratios.add_suffix('_c_predictions_set_')

    df = pd.concat([train_pg_ratios, predicted_pg_ratios, predictions_pg_ratios], axis=1)
    df = df.reindex(sorted(df.columns), axis=1)

    return df


def performance_assessment(data: Dataset, predictions: Dataset) -> pd.DataFrame:
    performance_evaluator = ModelEvaluator(data, predictions)
    return performance_evaluator.evaluate().reset_index(drop=True)


def classifier_assessment(classifier: object, train_data: Dataset, validation_data: Dataset,
                          protected_attribute: str = 'NA') -> (pd.Series, pd.DataFrame):

    predictions = get_classifier_predictions(classifier, train_data, validation_data)

    fairness_metrics = fairness_assessment(validation_data, predictions, protected_attribute)
    performance_metrics = performance_assessment(validation_data, predictions)
    distribution = distribution_assessment(train_data, validation_data, predictions, protected_attribute)

    rows = max(len(fairness_metrics), len(performance_metrics))
    classifier_df = dict_to_dataframe({'classification_algorithm': classifier.__class__.__name__})
    metrics_info_df = pd.concat([classifier_df] * rows, ignore_index=True)
    metrics = pd.concat([pd.concat([metrics_info_df, fairness_metrics], axis=1), performance_metrics], axis=1)

    distribution = pd.concat([pd.concat([classifier_df] * len(distribution), ignore_index=True), distribution], axis=1)

    return predictions.targets, metrics, distribution
