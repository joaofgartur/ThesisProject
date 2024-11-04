"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, \
    matthews_corrcoef, roc_curve, auc

from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from utils import dict_to_dataframe


class ModelEvaluator(object):
    """
    Class to evaluate model performance metrics.

    Attributes
    ----------
    y : pd.Series
        The true target values.
    y_pred : pd.Series
        The predicted target values.

    Methods
    -------
    __init__(targets: pd.DataFrame, predictions: pd.DataFrame):
        Initializes the ModelEvaluator with the provided targets and predictions.
    confusion_matrix():
        Computes the confusion matrix.
    roc_curve():
        Computes the ROC curve.
    auc():
        Computes the area under the ROC curve.
    accuracy():
        Computes the accuracy score.
    f1_score():
        Computes the F1 score.
    recall():
        Computes the recall score.
    precision():
        Computes the precision score.
    mathews_cc():
        Computes the Matthews correlation coefficient.
    sensitivity():
        Computes the sensitivity (recall for the positive class).
    specificity():
        Computes the specificity (recall for the negative class).
    evaluate():
        Evaluates and returns selected performance metrics as a DataFrame.
    """

    def __init__(self, targets: pd.DataFrame, predictions: pd.DataFrame):
        """
        Initializes the ModelEvaluator with the provided targets and predictions.

        Parameters
        ----------
        targets : pd.DataFrame
            The true target values.
        predictions : pd.DataFrame
            The predicted target values.
        """
        self.y = targets.squeeze()
        self.y_pred = predictions.squeeze()

    def confusion_matrix(self):
        """
        Computes the confusion matrix.

        Returns
        -------
        list
            The confusion matrix as a flattened list.
        """
        return [list(np.concatenate(confusion_matrix(y_true=self.y, y_pred=self.y_pred)).flat)]

    def roc_curve(self):
        """
        Computes the ROC curve.

        Returns
        -------
        list
            The false positive rates, true positive rates, and thresholds.
        """
        fpr, tpr, thresholds = roc_curve(self.y, self.y_pred)
        return [[list(fpr), list(tpr), list(thresholds)]]

    def auc(self):
        """
        Computes the area under the ROC curve.

        Returns
        -------
        float
            The area under the ROC curve.
        """
        fpr, tpr, _ = roc_curve(self.y, self.y_pred)
        return auc(fpr, tpr)

    def accuracy(self):
        """
        Computes the accuracy score.

        Returns
        -------
        float
            The accuracy score.
        """
        return accuracy_score(y_true=self.y, y_pred=self.y_pred)

    def f1_score(self):
        """
        Computes the F1 score.

        Returns
        -------
        float
            The F1 score.
        """
        return f1_score(y_true=self.y, y_pred=self.y_pred)

    def recall(self):
        """
        Computes the recall score.

        Returns
        -------
        float
            The recall score.
        """
        return recall_score(y_true=self.y, y_pred=self.y_pred)

    def precision(self):
        """
        Computes the precision score.

        Returns
        -------
        float
            The precision score.
        """
        return precision_score(y_true=self.y, y_pred=self.y_pred)

    def mathews_cc(self):
        """
        Computes the Matthews correlation coefficient.

        Returns
        -------
        float
            The Matthews correlation coefficient.
        """
        return matthews_corrcoef(y_true=self.y, y_pred=self.y_pred)

    def sensitivity(self):
        """
        Computes the sensitivity (recall for the positive class).

        Returns
        -------
        float
            The sensitivity score.
        """
        return recall_score(y_true=self.y, y_pred=self.y_pred, pos_label=POSITIVE_OUTCOME)

    def specificity(self):
        """
        Computes the specificity (recall for the negative class).

        Returns
        -------
        float
            The specificity score.
        """
        return recall_score(y_true=self.y, y_pred=self.y_pred, pos_label=NEGATIVE_OUTCOME)

    def evaluate(self):
        """
        Evaluates and returns selected performance metrics as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the evaluated performance metrics.
        """
        result = {
            'performance_accuracy': self.accuracy(),
            'performance_f1_score': self.f1_score(),
            'performance_auc': self.auc(),
        }
        return dict_to_dataframe(result)
