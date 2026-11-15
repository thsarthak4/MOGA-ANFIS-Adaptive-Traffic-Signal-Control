import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from controllers import AnfisController
from simulator import simulate


def make_bounds(b):
    xl, xu = b.copy(), b.copy()
    xl[0:5] = np.maximum(b[0:5] - 15, 0);    xu[0:5] = b[0:5] + 15
    xl[5:10] = b[5:10] * 0.5;                xu[5:10] = b[5:10] * 1.8
    xl[10:15] = np.maximum(b[10:15] - 2, 0); xu[10:15] = b[10:15] + 2
    xl[15:20] = b[15:20] * 0.5;              xu[15:20] = b[15:20] * 1.8
    xl[20:], xu[20:] = 0.0, 1.0
    return xl, xu


class SignalProblem(ElementwiseProblem):
    def __init__(self, base, scenarios, seeds, n_obj, T=1800):
        xl, xu = make_bounds(base)
        super().__init__(n_var=len(base), n_obj=n_obj, xl=xl, xu=xu)
        self.scenarios, self.seeds, self.n, self.T = scenarios, seeds, n_obj, T

    def _evaluate(self, x, out, *args, **kwargs):
        ctrl = AnfisController(x)
        rs = [simulate(ctrl, sc, s, self.T) for sc in self.scenarios for s in self.seeds]
        d = np.mean([r["delay"] for r in rs])
        q = np.mean([r["queue"] for r in rs])
        st = np.mean([r["stops"] for r in rs])
        out["F"] = [d, q, st] if self.n == 3 else [d]


def run_opt(base, scenarios, n_obj, pop=40, gen=30, seed=1, seeds=(1000, 1001), callback=None):
    """n_obj=1 -> single-objective GA (Base Paper 1). n_obj=3 -> NSGA-II (our MOGA-ANFIS)."""
    prob = SignalProblem(base, scenarios, seeds, n_obj)
    rng = np.random.default_rng(seed)
    X0 = base + rng.normal(0, 1, (pop, len(base))) * (prob.xu - prob.xl) * 0.05
    X0 = np.clip(X0, prob.xl, prob.xu); X0[0] = base            # seed population around pretrained ANFIS
    algo = (NSGA2 if n_obj > 1 else GA)(pop_size=pop, sampling=X0)
    res = minimize(prob, algo, ("n_gen", gen), seed=seed, verbose=False,
                   save_history=True, callback=callback)
    res.conv = [float(np.min(h.opt.get("F")[:, 0])) for h in res.history]   # best delay per generation
    return res


def pick_compromise(F):
    """Pareto solution closest to the ideal point after min-max normalisation."""
    F = np.atleast_2d(F)
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    return int(np.argmin(np.sqrt((Fn ** 2).sum(1))))