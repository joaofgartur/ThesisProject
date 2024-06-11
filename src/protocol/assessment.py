"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

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

    x_train = train_data.features.to_numpy()
    y_train = train_data.targets.to_numpy().ravel()

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(validation_data.features.to_numpy())
    predicted_data = update_dataset(validation_data, targets=predictions)

    return predicted_data


def fairness_assessment(data: Dataset, predictions: Dataset, sensitive_attribute: str) -> pd.DataFrame:
    original_attribute_values = data.protected_attributes[sensitive_attribute]

    dummy_values = data.get_dummy_protected_feature(sensitive_attribute)

    assessment_df = pd.DataFrame()
    for value in dummy_values:
        data.protected_attributes[sensitive_attribute] = dummy_values[value]
        fairness_evaluator = FairnessEvaluator(data, predictions, sensitive_attribute)
        value_df = pd.concat([dict_to_dataframe({'value': value}), fairness_evaluator.evaluate()], axis=1)
        assessment_df = pd.concat([assessment_df, value_df])

    data.protected_attributes[sensitive_attribute] = original_attribute_values

    return assessment_df.reset_index(drop=True)


def performance_assessment(data: Dataset, predictions: Dataset) -> pd.DataFrame:
    performance_evaluator = ModelEvaluator(data, predictions)
    return performance_evaluator.evaluate().reset_index(drop=True)


def classifier_assessment(classifier: object, train_data: Dataset, validation_data: Dataset,
                          protected_attribute: str = 'NA') -> (pd.Series, pd.DataFrame):

    predictions = get_classifier_predictions(classifier, train_data, validation_data)

    fairness_metrics = fairness_assessment(validation_data, predictions, protected_attribute)
    performance_metrics = performance_assessment(validation_data, predictions)

    rows = max(len(fairness_metrics), len(performance_metrics))
    classification_algorithm = pd.concat([dict_to_dataframe({'classification_algorithm': classifier.__class__.__name__})] * rows, ignore_index=True)

    results = concat_df(concat_df(classification_algorithm, fairness_metrics, axis=1), performance_metrics, axis=1)

    return predictions.targets, results


def data_assessment(original_data: Dataset, transformed_data: Dataset, sensitive_attribute: str):
    print(f' --------- Assessment for {bold(original_data.name)} --------- ')

    print('Classes:')
    print(f'Original Data: \n{original_data.targets.value_counts()}')
    print(f'Transformed Data: \n{transformed_data.targets.value_counts()}')

    print('Protected Attributes:')
    print(f'Mapping:\n {original_data.features_mapping[sensitive_attribute]}')
    print(f'Original Protected Attributes: \n{original_data.features[sensitive_attribute].value_counts()}')
    print(f'Transformed Protected Attributes: \n{original_data.features[sensitive_attribute].value_counts()}')

    print('Data Description')
    print(f'Original Data Description: \n{original_data.features.describe().to_string()}')
    print(f'Transformed Data Description: \n{transformed_data.features.describe().to_string()}')
