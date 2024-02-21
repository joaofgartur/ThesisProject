"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset
from classifiers.classifiers import train_classifier
from metrics import compute_metrics_suite
from errors import error_check_dataset, error_check_sensitive_attribute
from helpers import logger, set_dataset_labels


def assess_classifier(classifier: object, dataset: Dataset, sensitive_attribute: str, settings: dict):
    classifier_name = classifier.__class__.__name__

    # obtain classifier performance
    predictions, accuracy = train_classifier(dataset, classifier, settings)
    dataset = set_dataset_labels(dataset, predictions)

    # compute fairness metrics
    metrics = compute_metrics_suite(dataset, sensitive_attribute)

    # save results
    results = [classifier_name, accuracy]
    results += metrics.values()

    return results


def bias_assessment(dataset: Dataset, settings: dict, intervention_attribute: str = 'NA', algorithm: str = 'NA') \
        -> pd.DataFrame:
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

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        #"Support Vector Machine": SVC(),
        #"Naive Bayes": GaussianNB(),
        #"Stochastic Gradient": SGDClassifier(),
        #"K-Nearest Neighbours": KNeighborsClassifier(),
        #"Decision Tree": DecisionTreeClassifier(),
        #"Random Forest": RandomForestClassifier()
    }

    results = []
    for sensitive_attribute in dataset.protected_features:
        logger.info(f"Computing fairness metrics for attribute \'{sensitive_attribute}\'...")

        error_check_sensitive_attribute(dataset, sensitive_attribute)

        logger.info(f"Fairness metrics for attribute \'{sensitive_attribute}\' computed.")

        # assess classifier performance
        for classifier in classifiers:
            classifier_results = [dataset.name, sensitive_attribute, intervention_attribute,
                                  algorithm]
            classifier_results += assess_classifier(classifiers[classifier], dataset, sensitive_attribute, settings)
            results.append(classifier_results)

    results = pd.DataFrame(results, columns=['Dataset', 'Protected Attribute', 'Intervention Attribute', 'Algorithm',
                                             'Classifier', 'Accuracy', "Disparate Impact", "Discrimination Score"])

    return results
