"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset, update_dataset, match_features
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import bold, dict_to_dataframe
from evaluation import FairnessEvaluator

surrogate_models = {
    #'LR': LogisticRegression(),
    #'SVC': SVC(),
    #'GNB': GaussianNB(),
    #'KNN': KNeighborsClassifier(),
    #"DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier()
}


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


def performance_assessment(data: Dataset, predictions: Dataset) -> pd.DataFrame:
    performance_evaluator = ModelEvaluator(data, predictions)
    return performance_evaluator.evaluate().reset_index(drop=True)


def assess_model(model: object, train_data: Dataset, validation_data: Dataset, sensitive_attribute: str = 'NA'):
    predictions = get_model_predictions(model, train_data, validation_data)

    fairness_metrics = fairness_assessment(validation_data, predictions, sensitive_attribute)
    performance_metrics = performance_assessment(validation_data, predictions)

    model_name = dict_to_dataframe({'model': model.__class__.__name__})

    results = pd.concat([model_name, fairness_metrics, performance_metrics], axis=1)

    return predictions.targets, results


def assess_all_surrogates(train_set: Dataset,
                          validation_set: Dataset,
                          protected_feature: str = 'NA') -> pd.DataFrame:

    assessment_df = pd.DataFrame()

    for surrogate in surrogate_models:
        _, surrogate_model_results = assess_model(
            surrogate_models[surrogate],
            train_set,
            validation_set,
            protected_feature)

        assessment_df = pd.concat([assessment_df, surrogate_model_results])

    return assessment_df


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
