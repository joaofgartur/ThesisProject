"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import pandas as pd


def joint_probability(data: pd.DataFrame, variables: dict) -> float:
    probs = data.value_counts(list(variables.keys()), normalize=True)
    return probs[tuple(variables.values())] if tuple(variables.values()) in probs else 0.0
