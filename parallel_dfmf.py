__author__ = 'urosbajc'

import numpy as np
import time
from skfusion.fusion import ObjectType, Relation, FusionGraph, Dfmf
from joblib import Parallel, delayed
import time


def rmse(X, X_):
    """
    Calculates root mean squared error between two matrices X and X_

    :param X:
    :param X_:
    :return:
    """

    return np.sqrt(np.mean((X - X_) ** 2))


def run_parallel(n_jobs=1):

    start_t = time.time()
    n1, n2 = 500, 500

    # poizkusil tudi z:
    # n1, n2 = 10000, 10000
    R12 = np.random.rand(n1, n2)
    print(f"Number of jobs is {n_jobs}")

    r1, r2 = 10, 10
    # Poizkusil tudi z
    # r1, r2 = 150, 150
    t1 = ObjectType('type1', r1)
    t2 = ObjectType('type2', r2)
    relations = [Relation(R12, t1, t2)]
    fusion_graph = FusionGraph(relations)

    fuser = Dfmf(init_type='random_vcol', n_jobs=n_jobs).fuse(fusion_graph)
    preds = fuser.complete(relations[0])
    # print(f"Error is {rmse(R12, preds)}")

    print(f"Done in {time.time() - start_t} sec.")


if __name__ == "__main__":
    for n_jobs in [1, 2, 4]:
        run_parallel(n_jobs)

