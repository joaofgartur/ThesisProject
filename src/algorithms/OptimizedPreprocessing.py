from aif360.algorithms.preprocessing import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from helpers import convert_to_standard_dataset, logger
from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset


class OptimizedPreprocessing(Algorithm):

    def __init__(self, optimization_parameters: dict):
        super().__init__()
        self.optimization_parameters = optimization_parameters

    def repair(self, dataset: Dataset, sensitive_attribute: str):
        logger.info(f"Repairing dataset {dataset.name} via Massaging...")

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # transform dataset
        transformer = OptimPreproc(OptTools, self.optimization_parameters)
        transformer = transformer.fit(standard_dataset)
        transformed_dataset = transformer.transform(standard_dataset, transform_Y=True)

        # convert into regular dataset
        new_dataset = update_dataset(dataset, features=transformed_dataset.features, targets=transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
