import numpy as np

from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import convert_to_standard_dataset, split_dataset, concatenate_ndarrays, modify_dataset


class DisparateImpactRemover(Algorithm):

    def __init__(self, repair_level: float, learning_settings: dict):
        self.repair_level = repair_level
        self.learning_settings = learning_settings

    def repair(self, dataset: Dataset, sensitive_attribute: str):
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        scaler = MinMaxScaler(copy=False)
        train, test = split_dataset(standard_dataset, self.learning_settings["train_size"])
        train.features = scaler.fit_transform(train.features)
        test.features = scaler.fit_transform(test.features)

        index = train.feature_names.index(sensitive_attribute)

        di = DIR(repair_level=self.repair_level)
        train_repd = di.fit_transform(train)
        test_repd = di.fit_transform(test)

        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features, index, axis=1)
        y_tr = train_repd.labels.ravel()

        lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
        lmod.fit(X_tr, y_tr)

        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = lmod.predict(X_te)
        test_repd_pred.labels.resize((len(test_repd_pred.labels), 1))

        features = concatenate_ndarrays(train_repd.features, test_repd_pred.features)
        labels = concatenate_ndarrays(train_repd.labels, test_repd_pred.labels)
        new_dataset = modify_dataset(dataset, features, labels)

        return new_dataset
