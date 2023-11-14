from datasets import Dataset


def pre_correction_diagnostics(dataset: Dataset):
    raise NotImplementedError


def post_pre_correction_diagnostics(pre_correction_diagnostics: dict, dataset: Dataset):
    raise NotImplementedError
