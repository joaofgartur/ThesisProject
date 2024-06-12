import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, \
    matthews_corrcoef, roc_curve, auc

from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset
from helpers import dict_to_dataframe


class ModelEvaluator(object):

    def __init__(self, original_data: Dataset, predicted_data: Dataset):
        self.y = original_data.targets.squeeze()
        self.y_pred = predicted_data.targets.squeeze()

    def confusion_matrix(self):
        return [list(np.concatenate(confusion_matrix(y_true=self.y, y_pred=self.y_pred)).flat)]

    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y, self.y_pred)
        return [[list(fpr), list(tpr), list(thresholds)]]

    def auc(self):
        fpr, tpr, _ = roc_curve(self.y, self.y_pred)
        return auc(fpr, tpr)

    def accuracy(self):
        return accuracy_score(y_true=self.y, y_pred=self.y_pred)

    def f1_score(self):
        return f1_score(y_true=self.y, y_pred=self.y_pred)

    def recall(self):
        return recall_score(y_true=self.y, y_pred=self.y_pred)

    def precision(self):
        return precision_score(y_true=self.y, y_pred=self.y_pred)

    def mathews_cc(self):
        return matthews_corrcoef(y_true=self.y, y_pred=self.y_pred)

    def sensitivity(self):
        return recall_score(y_true=self.y, y_pred=self.y_pred, pos_label=POSITIVE_OUTCOME)

    def specificity(self):
        return recall_score(y_true=self.y, y_pred=self.y_pred, pos_label=NEGATIVE_OUTCOME)

    def evaluate(self):
        result = {
            'performance_accuracy': self.accuracy(),
            'performance_f1_score': self.f1_score(),
            'performance_auc': self.auc(),
        }
        return dict_to_dataframe(result)
