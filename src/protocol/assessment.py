"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""
import numpy as np
import pandas as pd


from constants import PRED_OUTCOME
from datasets import Dataset
from evaluation.ModelEvaluator import ModelEvaluator
from utils import dict_to_dataframe
from evaluation import FairnessEvaluator


def get_classifier_predictions(model: object, train: Dataset, validation: Dataset) -> pd.DataFrame:
    """
    Get predictions from a classifier model.

    Parameters
    ----------
    model : object
        The classifier model.
    train : Dataset
        The training dataset.
    validation : Dataset
        The validation dataset.

    Returns
    -------
    pd.DataFrame
        The predictions as a DataFrame.
    """

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('classifier', model)
    ], memory=None)

    x_train = train.features.copy()
    x_val = validation.features.copy()

    # match features
    common = x_train.columns.intersection(x_val.columns)
    x_val = x_val.drop(columns=x_val.columns.difference(common))

    x_train = x_train.to_numpy().astype(np.float32)
    y_train = train.targets.to_numpy().astype(np.float32).ravel()

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_val.to_numpy())
    predictions = pd.DataFrame(predictions, columns=[PRED_OUTCOME])

    return predictions


def fairness_assessment(data: Dataset, predictions: pd.DataFrame, sensitive_attribute: str) -> pd.DataFrame:
    """
    Assess fairness metrics for the given data and predictions.

    Parameters
    ----------
    data : Dataset
        The dataset.
    predictions : pd.DataFrame
        The predictions.
    sensitive_attribute : str
        The sensitive attribute.

    Returns
    -------
    pd.DataFrame
        The fairness metrics as a DataFrame.
    """

    dummy_values = data.get_dummy_protected_feature(sensitive_attribute)
    metrics = pd.DataFrame()
    for value in dummy_values:
        df = pd.DataFrame(dummy_values[value], columns=[value])
        fairness_evaluator = FairnessEvaluator(data.features, data.targets, predictions, df)
        value_metrics = pd.concat([dict_to_dataframe({'value': value}), fairness_evaluator.evaluate()], axis=1)
        metrics = pd.concat([metrics, value_metrics])

    return metrics.reset_index(drop=True)


def distribution_assessment(train_data: Dataset, predicted_data: Dataset, predictions: pd.DataFrame, protected_attribute: str = 'NA') -> pd.DataFrame:
    """
    Assess the distribution of the protected attribute in the training, predicted, and prediction sets.

    Parameters
    ----------
    train_data : Dataset
        The training dataset.
    predicted_data : Dataset
        The predicted dataset.
    predictions : pd.DataFrame
        The predictions.
    protected_attribute : str, optional
        The protected attribute (default is 'NA').

    Returns
    -------
    pd.DataFrame
        The distribution metrics as a DataFrame.
    """

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

    predicted_s = predicted_data.protected_features[protected_attribute].apply(map_protected_attribute)
    predicted_pg = pd.concat([predicted_s, predicted_data.targets], axis=1)
    predicted_pg_ratios = get_value_counts(predicted_pg)
    predicted_pg_ratios = predicted_pg_ratios.add_suffix('_b_predicted_set_')

    predictions_pg = pd.concat([predicted_s, predictions * 1.0], axis=1)
    predictions_pg_ratios = get_value_counts(predictions_pg)
    predictions_pg_ratios = predictions_pg_ratios.add_suffix('_c_predictions_set_')

    df = pd.concat([train_pg_ratios, predicted_pg_ratios, predictions_pg_ratios], axis=1)
    df = df.reindex(sorted(df.columns), axis=1)

    return df


def classifier_assessment(classifier: object, train_data: Dataset, validation_data: Dataset,
                          sensitive_attribute: str = 'NA') -> (pd.Series, pd.DataFrame):
    """
    Assess the classifier's performance and fairness.

    Parameters
    ----------
    classifier : object
        The classifier model.
    train_data : Dataset
        The training dataset.
    validation_data : Dataset
        The validation dataset.
    sensitive_attribute : str, optional
        The sensitive attribute (default is 'NA').

    Returns
    -------
    pd.Series
        The predictions.
    pd.DataFrame
        The combined metrics and distribution as a DataFrame.
    """

    predictions = get_classifier_predictions(classifier, train_data, validation_data)

    # fairness metrics

    fairness_metrics = fairness_assessment(validation_data, predictions, sensitive_attribute)
    n_rows = fairness_metrics.shape[0]

    # performance metrics
    evaluator = ModelEvaluator(validation_data.targets, predictions)
    performance_metrics = evaluator.evaluate().reset_index(drop=True)
    performance_metrics = pd.concat([performance_metrics] * n_rows, ignore_index=True)

    # metadata
    metadata = dict_to_dataframe({'classification_algorithm': classifier.__class__.__name__})
    n_rows_metadata = pd.concat([metadata] * n_rows, ignore_index=True)

    metrics = pd.concat([n_rows_metadata, fairness_metrics, performance_metrics], axis=1)

    # distribution
    distribution = distribution_assessment(train_data, validation_data, predictions, sensitive_attribute)
    distribution = pd.concat([pd.concat([metadata] * distribution.shape[0], ignore_index=True), distribution], axis=1)

    return predictions, metrics, distribution
