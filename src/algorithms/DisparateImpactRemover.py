import copy
import numpy as np
import pandas as pd

from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR
from aif360.datasets import StandardDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from constants import POSITIVE_OUTCOME, PRIVILEGED
from datasets import Dataset


def convert_to_aif_dataset(dataset: Dataset, sensitive_attribute: str):
    dataframe, label_name = dataset.merge_features_and_targets()
    favorable_classes = [POSITIVE_OUTCOME]
    privileged_classes = [PRIVILEGED]
    features_to_keep = dataset.features.columns.tolist()
    aif_dataset = StandardDataset(df=dataframe,
                                  label_name=label_name,
                                  favorable_classes=favorable_classes,
                                  privileged_classes=[privileged_classes],
                                  features_to_keep=features_to_keep,
                                  protected_attribute_names=[sensitive_attribute])

    return aif_dataset


class DisparateImpactRemover:

    def __init__(self, repair_level: float):
        self.repair_level = repair_level

    def repair(self, dataset: Dataset, sensitive_attribute: str, learning_settings: dict):
        new_dataset = copy.deepcopy(dataset)
        df = convert_to_aif_dataset(dataset, sensitive_attribute)

        scaler = MinMaxScaler(copy=False)
        test, train = df.split([learning_settings["test_size"]])
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

        df.features = np.concatenate((train_repd.features, test_repd_pred.features), axis=0)
        df.labels = np.concatenate((train_repd.labels, test_repd_pred.labels), axis=0)

        new_dataset.features = pd.DataFrame(df.features, columns=dataset.features.columns)
        new_dataset.targets = pd.DataFrame(df.labels, columns=dataset.targets.columns)

        return new_dataset
