from helpers import logger


def extract_value(key: str, dictionary: dict):
    if key in dictionary.keys():
        return dictionary[key]
    logger.error(f'Missing information: {key}')
    raise ValueError
