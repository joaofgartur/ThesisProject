import pandas as pd


def conditional_probability(data: pd.DataFrame, A, a, B, b):
    """P(A=a|B=b)"""
    return joint_probability(data, A, a, B, b) / simple_probability(data, B, b)


def joint_probability(data: pd.DataFrame, A, a, B, b):
    """P(A=a & B=b)"""
    n_ab = data[(data[A] == a) & (data[B] == b)].shape[0]
    n = data.shape[0]
    return n_ab / n


def simple_probability(data: pd.DataFrame, A, a):
    """P(A=a)"""
    n_a = data[data[A] == a].shape[0]
    n = data.shape[0]
    return n_a / n
