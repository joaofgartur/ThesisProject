import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, matthews_corrcoef

from constants import NUM_DECIMALS, POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset
from helpers import dict_to_dataframe


class ModelEvaluator(object):

    def __init__(self, original_data: Dataset, predicted_data: Dataset):
        self.y = original_data.targets.squeeze()
        self.y_pred = predicted_data.targets.squeeze()

    def confusion_matrix(self):
        return [list(np.concatenate(confusion_matrix(y_true=self.y, y_pred=self.y_pred)).flat)]

    def accuracy(self):
        return np.round(accuracy_score(y_true=self.y, y_pred=self.y_pred), decimals=NUM_DECIMALS)

    def f1_score(self):
        return np.round(f1_score(y_true=self.y, y_pred=self.y_pred), decimals=NUM_DECIMALS)

    def recall(self):
        return np.round(recall_score(y_true=self.y, y_pred=self.y_pred), decimals=NUM_DECIMALS)

    def precision(self):
        return np.round(precision_score(y_true=self.y, y_pred=self.y_pred), decimals=NUM_DECIMALS)

    def mathews_cc(self):
        return np.round(matthews_corrcoef(y_true=self.y, y_pred=self.y_pred), decimals=NUM_DECIMALS)

    def sensitivity(self):
        return np.round(recall_score(y_true=self.y, y_pred=self.y_pred, pos_label=POSITIVE_OUTCOME), decimals=NUM_DECIMALS)

    def specificity(self):
        return np.round(recall_score(y_true=self.y, y_pred=self.y_pred, pos_label=NEGATIVE_OUTCOME),
                        decimals=NUM_DECIMALS)

    def evaluate(self):
        result = {
            'confusion_matrix': self.confusion_matrix(),
            'accuracy': self.accuracy(),
            'f1_score': self.f1_score(),
            'recall': self.recall(),
            'precision': self.precision(),
            'mathews_cc': self.mathews_cc(),
            'sensitivity': self.sensitivity(),
            'specificity': self.specificity()
        }
        return dict_to_dataframe(result)
