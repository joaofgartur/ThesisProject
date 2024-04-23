import copy

import numpy as np
import scipy.optimize as optim

from algorithms.LFRhelpers import LFR, set_numba_seed
from constants import UNPRIVILEGED, PRIVILEGED
from datasets import Dataset, update_dataset

from algorithms.Algorithm import Algorithm
from helpers import logger


class LearnedFairRepresentations(Algorithm):

    def __init__(self,
                 k: int = 5,
                 ax: float = 0.01,
                 ay: float = 1.0,
                 az: float = 50.0,
                 seed: int = None,
                 max_func_evals: int = 150000,
                 max_iterations: int = 150000):
        super().__init__()

        # define jit seed
        set_numba_seed(seed)

        self.k = k
        self.Ax = ax
        self.Ay = ay
        self.Az = az

        self.max_func_evals = max_func_evals
        self.max_iterations = max_iterations
        self.optimization_results = None

    def __optimization_parameters_init(self, k: int, n_dimensions: int):
        initial_parameters = np.random.uniform(size=2 * n_dimensions + k * n_dimensions + k).astype(np.float64)

        bounds = []
        for i, k2 in enumerate(initial_parameters):
            if i < n_dimensions * 2 or i >= n_dimensions * 2 + k:
                bounds.append((None, None))
            else:
                bounds.append((0, 1))

        return initial_parameters, bounds

    def __extract_subgroup(self, data: Dataset, protected_attribute: str, privilege: int):
        subgroup_indexes = data.features.loc[data.features[protected_attribute] == privilege].index
        x_subgroup = data.features.loc[subgroup_indexes].values
        y_subgroup = data.targets.loc[subgroup_indexes].values

        return subgroup_indexes, x_subgroup, y_subgroup

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute

        _, x_privileged, y_privileged = self.__extract_subgroup(data, sensitive_attribute, PRIVILEGED)
        _, x_unprivileged, y_unprivileged = self.__extract_subgroup(data, sensitive_attribute, UNPRIVILEGED)

        _, n_dimensions = data.features.shape
        initial_parameters, bounds = self.__optimization_parameters_init(self.k, n_dimensions)

        self.optimization_results = optim.fmin_l_bfgs_b(LFR,
                                                        x0=initial_parameters,
                                                        epsilon=1e-5,
                                                        args=(x_unprivileged, x_privileged, y_unprivileged, y_privileged,
                                                    self.k, self.Ax, self.Ay, self.Az),
                                                        bounds=bounds,
                                                        approx_grad=True,
                                                        maxfun=self.max_func_evals,
                                                        maxiter=self.max_iterations)

        if self.optimization_results[2]['warnflag'] != 0:
            logger.warning("Optimization did not converge. The message is: %s" % self.optimization_results[2]['task'])

    def transform(self, data: Dataset):

        privileged_indexes, x_privileged, y_privileged = self.__extract_subgroup(data, self.sensitive_attribute, PRIVILEGED)
        unprivileged_indexes, x_unprivileged, y_unprivileged = self.__extract_subgroup(data, self.sensitive_attribute, UNPRIVILEGED)

        transformed_y_unprivileged, transformed_y_privileged, _, _, transformed_m_unprivileged, transformed_m_privileged = LFR(
            self.optimization_results[0],
            x_unprivileged,
            x_privileged,
            y_unprivileged,
            y_privileged,
            self.k,
            self.Ax,
            self.Ay,
            self.Az,
            1)

        transformed_features = np.zeros(shape=data.features.shape)
        transformed_targets = np.zeros(shape=data.targets.shape)

        transformed_features[privileged_indexes] = transformed_m_privileged
        transformed_targets[privileged_indexes] = np.reshape(transformed_y_privileged, [-1, 1])

        transformed_features[unprivileged_indexes] = transformed_m_unprivileged
        transformed_targets[unprivileged_indexes] = np.reshape(transformed_y_unprivileged, [-1, 1])

        transformed_targets = (np.array(transformed_targets) > 0.5).astype(np.float64)

        unique_values = np.unique(transformed_targets)
        if len(unique_values) == 1 and (unique_values[0] == 0 or unique_values[0] == 1):
            print('All targets are either 0 or 1. This is not expected. Check the optimization results.')
            return data

        transformed_dataset = copy.deepcopy(data)
        transformed_dataset = update_dataset(dataset=transformed_dataset,
                                             features=transformed_features,
                                             targets=transformed_targets)

        return transformed_dataset
