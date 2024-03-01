from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import logger
from .correction import bias_correction


class Pipeline:

    def __init__(self, dataset: Dataset, algorithm: Algorithm, settings: dict) -> None:
        self.dataset = dataset
        self.algorithm = algorithm
        self.settings = settings

    def run(self) -> None:
        try:
            train_set, test_set = self.dataset.split(self.settings)

            bias_correction(train_set, self.settings, [self.algorithm])



        except Exception as e:
            logger.error(f'An error occurred in the pipeline: \n {e}')
            raise
