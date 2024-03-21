import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, matthews_corrcoef

from constants import NUM_DECIMALS
from datasets import Dataset


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

    def evaluate(self):
        return {
            'confusion_matrix': self.confusion_matrix(),
            'accuracy': self.accuracy(),
            'f1_score': self.f1_score(),
            'recall': self.recall(),
            'precision': self.precision(),
            'mathews_cc': self.mathews_cc()
        }
