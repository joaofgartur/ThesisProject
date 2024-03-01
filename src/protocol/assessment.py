"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import pandas as pd
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

from datasets import Dataset
from errors import error_check_dataset, error_check_sensitive_attribute
from helpers import logger


def assess_surrogate_model(model: object, dataset: Dataset, sensitive_attribute: str, settings: dict):
    classifier_name = model.__class__.__name__

    # split into train and validation sets
    train_data, validation_data = dataset.split(settings, test=False)

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model)
    ])

    x_train = train_data.features
    y_train = train_data.targets.to_numpy().ravel()

    pipeline.fit(x_train, y_train)

    y_val = validation_data.targets.to_numpy().ravel()
    predictions = pipeline.predict(validation_data.features)
    accuracy = accuracy_score(y_val, predictions.ravel())

    # obtain classifier performance
    # dataset = set_dataset_labels(dataset, predictions)

    # compute fairness metrics
    # metrics = compute_metrics_suite(dataset, sensitive_attribute)

    # save results
    results = [classifier_name, accuracy]
    # results += metrics.values()

    return results


def assess_all_surrogates(dataset: Dataset, settings: dict, intervention_attribute: str = 'NA', algorithm: str = 'NA'):
    """
    Conduct assessment on a dataset, including fairness metrics and classifier accuracies.

    Parameters
    ----------
    algorithm
    intervention_attribute
    dataset : Dataset
        The dataset object containing features, targets, and sensitive attributes.
    settings : dict
        Dictionary containing learning settings.

    Returns
    -------
    pd.DataFrame
        A dictionary containing fairness metrics and classifier accuracies.

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
        - If there are missing sensitive attributes in the dataset.
        - If learning settings are missing required keys.
    """
    error_check_dataset(dataset)

    surrogate_classifiers = {
        'LR': LogisticRegression(),
        #'SVC': SVC(),
        #'GNB': GaussianNB(),
        #'KNN': KNeighborsClassifier(),
        #"DT": DecisionTreeClassifier(),
        #"RF": RandomForestClassifier()
    }

    global_results = []
    for feature in dataset.protected_features:
        error_check_sensitive_attribute(dataset, feature)

        for surrogate in surrogate_classifiers:
            logger.info(f'Assessing surrogate {surrogate} for feature \"{feature}\".')

            local_results = [dataset.name, feature, intervention_attribute, algorithm]
            local_results += assess_surrogate_model(surrogate_classifiers[surrogate], dataset, feature, settings)

            print(local_results)
            # global_results.append(local_results)

    # global_results = pd.DataFrame(global_results, columns=['Dataset', 'Protected Attribute', 'Intervention Attribute', 'Algorithm',
                                             # 'Classifier', 'Accuracy', "Disparate Impact", "Discrimination Score"])

    return global_results
