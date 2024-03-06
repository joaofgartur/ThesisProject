from helpers import logger


def bold(string: str) -> str:
    bold_start = "\033[1m"
    bold_end = "\033[0m"

    return bold_start + string + bold_end


def extract_value(key: str, dictionary: dict):
    if key in dictionary.keys():
        return dictionary[key]
    logger.error(f'Missing information: {key}')
    raise ValueError


def safe_division(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0
