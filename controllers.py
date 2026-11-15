import numpy as np
from anfis import ANFIS, QL_MAX, V_MAX
from simulator import BASE_DEMAND, LANES, SAT, LOST


def webster_greens(demand):
    """Webster: C0=(1.5L+5)/(1-Y), ge_i=(C0-L)*y_i/Y. Returns (greens, cycle)."""
    y = np.asarray(demand) / (SAT * LANES)
    L = LOST * 3
    Y = min(y.sum(), 0.85)
    C = (1.5 * L + 5) / (1 - Y)
    g = np.maximum((C - L) * y / y.sum(), 10.0)
    return g, C


GMAX_FACTOR = 1.6
GMIN = 15.0
GMAX = GMAX_FACTOR * webster_greens(BASE_DEMAND)[0]     # max green per phase


class FixedTime:
    def __init__(self, greens=(40, 22, 22), name="Fixed-time"):
        self.greens, self.name = list(greens), name
    def green(self, p, ql, v):
        return self.greens[p]


class WebsterController(FixedTime):
    def __init__(self, greens, name="Webster"):
        super().__init__(greens, name)


class AnfisController:
    """Green = gmin + U*(gmax-gmin), with U from the ANFIS (paper eq. 10)."""
    def __init__(self, params, name="ANFIS"):
        self.model, self.name = ANFIS(params), name
    def green(self, p, ql, v):
        u = self.model.predict(np.array([min(ql, QL_MAX)]), np.array([min(v, V_MAX)]))[0]
        return GMIN + u * (GMAX[p] - GMIN)