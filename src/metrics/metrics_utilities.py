import pandas as pd
import numpy as np


def conditional_probability(data: pd.DataFrame, A, a, B, b):
    return np.mean(data.query(B + "==" + str(b))[A] == a)


def simple_probability(data: pd.DataFrame, A, a):
    """P(A=a)"""
    return np.mean(data[A == a])
