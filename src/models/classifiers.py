"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
from typing import Any

from sklearn.metrics import accuracy_score

from datasets import Dataset
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler

from errors import error_check_dataset, error_check_dictionary_keys


def train_model(model: object, train_data: Dataset, settings: dict):
    """
    Train a classifier on the specified dataset and return the mean test score.

    Parameters
    ----------
    train_data :
        Dataset object containing features and targets.
    model :
        Classifier to be trained. It should be a callable function compatible with scikit-learn's API.
    settings : dict
        Dictionary containing learning settings, including 'test_size' and 'train_size' for train-test split.

    Returns
    -------
    float
        Mean test score of the trained classifier.

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
        - If 'test_size' and 'train_size' are not provided in the learning settings.
    """
    pass
    """
    error_check_dataset(train_data)

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model)
    ])

    pipeline.fit(train_data.features, train_data.targets)

    val_pred_set = pipeline.predict(val_set.features)

    accuracy = accuracy_score(val_set.targets, val_pred_set)

    return predictions

    print(train_set)
    print(val_set)

    x_train, __, y_train, __ = train_test_split(dataset.features, dataset.targets,
                                                test_size=learning_settings[_TEST_SIZE_KEY],
                                                train_size=learning_settings[_TRAIN_SIZE_KEY])

    if dataset.instance_weights is not None:
        train_sample_weights = dataset.get_train_sample_weights(x_train)
        predictions = cross_val_predict(pipeline, x_train, y_train.values.ravel(),
                                        cv=learning_settings[_CROSS_VALIDATION],
                                        fit_params={'classifier__sample_weight': train_sample_weights})
    else:
        predictions = cross_val_predict(pipeline, x_train, y_train.values.ravel(),
                                        cv=learning_settings[_CROSS_VALIDATION])

    # Compute mean test score
    scores = cross_validate(pipeline, x_train, y_train.values.ravel())
    accuracy = scores['test_score'].mean()

    return predictions, accuracy

    """
