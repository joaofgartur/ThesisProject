from datetime import datetime

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR
from aif360.datasets import BinaryLabelDataset
from matplotlib import pyplot as plt

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import convert_to_standard_dataset, set_dataset_features_and_labels, logger
import seaborn as sns


class DisparateImpactRemover(Algorithm):

    def __init__(self, repair_level: float):
        super().__init__()
        self.repair_level = repair_level

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        logger.info(f"Repairing dataset {dataset.name} via {self.__class__.__name__}...")

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        dataframe, label_name = dataset.merge_features_and_targets()
        train_BLD = BinaryLabelDataset(favorable_label='1',
                                       unfavorable_label='0',
                                       df=dataframe,
                                       label_names=[label_name],
                                       protected_attribute_names=[sensitive_attribute],
                                       unprivileged_protected_attributes=['0'])

        di = DIR(repair_level=1.0)
        rp_train = di.fit_transform(train_BLD)

        columns = train_BLD.feature_names + train_BLD.label_names
        print(columns)

        print(train_BLD.features.shape)
        print(train_BLD.labels.shape)

        rp_train_pd = pd.DataFrame(np.hstack([rp_train.features, rp_train.labels]),
                                   columns=columns)

        target_0_rep1 = rp_train_pd.loc[rp_train_pd[sensitive_attribute] == 0]
        target_1_rep1 = rp_train_pd.loc[rp_train_pd[sensitive_attribute] == 1]

        sns.distplot(target_0_rep1[label_name], hist=True, rug=False)
        sns.distplot(target_1_rep1[[label_name]], hist=True, rug=False)

        c = datetime.now()
        time = c.strftime('%d_%m_%y-%H_%M_%S')

        filename = time + f'-{dataset.name}-{self.__class__.__name__}.png'

        plt.show()
        plt.savefig(filename)

        # transform dataset
        transformer = DIR(repair_level=self.repair_level)
        transformed_dataset = transformer.fit_transform(standard_dataset)

        # convert into regular dataset
        new_dataset = set_dataset_features_and_labels(dataset=dataset,
                                                      features=transformed_dataset.features,
                                                      labels=transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
