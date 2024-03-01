from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import logger, write_dataframe_to_csv
from .correction import bias_correction


class Pipeline:

    def __init__(self, dataset: Dataset, algorithm: Algorithm, settings: dict) -> None:
        self.dataset = dataset
        self.algorithm = algorithm
        self.settings = settings
        self.results = None

    def run(self) -> None:

        try:
            train_set, test_set = self.dataset.split(self.settings)

            self.results = bias_correction(train_set, self.settings, [self.algorithm])

        except Exception as e:
            logger.error(f'An error occurred in the pipeline: \n {e}')
            raise

    def save(self, path: str = 'results') -> None:
        write_dataframe_to_csv(df=self.results, dataset_name=self.dataset.name,
                               path=path)

    def run_and_save(self, path: str = 'results') -> None:
        self.run()
        self.save(path)
