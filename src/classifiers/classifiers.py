"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
from typing import Tuple, Any

from datasets import Dataset
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler

from errors import error_check_dataset, error_check_dictionary_keys


def train_classifier(dataset: Dataset, classifier: object, learning_settings: dict) -> float:
    """
    Train a classifier on the specified dataset and return the mean test score.

    Parameters
    ----------
    dataset :
        Dataset object containing features and targets.
    classifier :
        Classifier to be trained. It should be a callable function compatible with scikit-learn's API.
    learning_settings : dict
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
    _TRAIN_SIZE_KEY = "test_size"
    _TEST_SIZE_KEY = "train_size"
    _CROSS_VALIDATION = 'cross_validation'

    error_check_dataset(dataset)
    error_check_dictionary_keys(learning_settings, [_TRAIN_SIZE_KEY, _TEST_SIZE_KEY])

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('clf', classifier)
    ])

    x_train, __, y_train, __ = train_test_split(dataset.features, dataset.targets,
                                                        test_size=learning_settings[_TEST_SIZE_KEY],
                                                        train_size=learning_settings[_TRAIN_SIZE_KEY])

    # Compute mean test score
    scores = cross_validate(pipeline, x_train, y_train.values.ravel())
    accuracy = scores['test_score'].mean()

    return accuracy


def train_all_classifiers(dataset: Dataset, learning_settings: dict) -> dict:
    """
    Train multiple classifiers and return their accuracy scores.

    Parameters
    ----------
    dataset :
        The dataset object containing features and targets.
    learning_settings :
        Dictionary containing learning settings.

    Returns
    -------
    dict
        A dictionary where keys are classifier names, and values are their accuracy scores.

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
        - If learning settings are missing required keys.
    """
    error_check_dataset(dataset)

    results = {}

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB(),
        "Stochastic Gradient": SGDClassifier(),
        "K-Nearest Neighbours": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    for classifier in classifiers:
        accuracy = train_classifier(dataset, classifiers[classifier], learning_settings)
        results.update({classifier: accuracy})

    return results
