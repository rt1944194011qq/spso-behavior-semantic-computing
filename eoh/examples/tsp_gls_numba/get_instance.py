import os
import pickle as pkl

import numpy as np


class GetData:
    """Loads pre-generated TSP instances with known optimal tour costs.

    The training set (TSPAEL64.pkl) holds 64 random 100-node Euclidean TSP
    instances together with their optimal tour costs, used to compute the
    optimality gap during evolution.
    """

    def __init__(self, n_instance, datafile=None):
        self.n_instance = n_instance
        if datafile is None:
            datafile = os.path.join(os.path.dirname(__file__),
                                    'TrainingData', 'TSPAEL64.pkl')
        self.datafile = os.path.abspath(os.fspath(datafile))

    def load_instances(self):
        with open(self.datafile, 'rb') as f:
            data = pkl.load(f)
        required = {'coordinate', 'optimal_tour', 'distance_matrix', 'cost'}
        missing = required.difference(data)
        if missing:
            raise ValueError(
                f"training data {self.datafile} is missing fields: {sorted(missing)}"
            )
        lengths = {key: len(data[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"training data fields have inconsistent lengths: {lengths}"
            )
        available = lengths['cost']
        if not 1 <= self.n_instance <= available:
            raise ValueError(
                f"requested {self.n_instance} instances, but {self.datafile} "
                f"contains {available}"
            )
        coords = [np.asarray(c) for c in data['coordinate'][:self.n_instance]]
        instances = [np.asarray(d) for d in data['distance_matrix'][:self.n_instance]]
        opt_costs = list(data['cost'][:self.n_instance])
        if any(cost is None or not np.isfinite(float(cost)) or float(cost) <= 0
               for cost in opt_costs):
            raise ValueError(
                f"the first {self.n_instance} instances in {self.datafile} "
                "must all have finite positive solved costs"
            )
        return coords, instances, opt_costs
