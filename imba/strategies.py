from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import math
import numpy as np

try:
    from hyperopt import fmin, tpe, Trials, STATUS_OK
except Exception:
    fmin = None

@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]]

class BayesianSearch:
    def __init__(self, space: Dict[str, Any], objective: Callable[[Dict[str, Any]], float], max_evals: int = 50, seed: int = 42):
        if fmin is None:
            raise ImportError("hyperopt is required for BayesianSearch")
        self.space, self.objective, self.max_evals, self.seed = space, objective, max_evals, seed

    def run(self) -> SearchResult:
        trials = Trials()
        def _obj(p):
            s = self.objective(p)
            return {"loss": -s, "status": STATUS_OK}
        rng = np.random.default_rng(self.seed)
        best = fmin(fn=_obj, space=self.space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials, rstate=rng)
        hist = []
        for t in trials.trials:
            vals = {k: (v[0] if isinstance(v, list) and v else v) for k, v in t["misc"]["vals"].items()}
            hist.append({"params": vals, "score": -t["result"]["loss"]})
        best_score = max((h["score"] for h in hist), default=float("nan"))
        return SearchResult(best_params=best, best_score=best_score, trials=hist)

class BanditUCB1:
    def __init__(self, arms: List[Dict[str, Any]], objective_arm: Callable[[Dict[str, Any]], float], budget: int = 60):
        self.arms, self.objective_arm, self.budget = arms, objective_arm, budget
        self.counts = [0] * len(arms)
        self.values = [0.0] * len(arms)

    def run(self) -> SearchResult:
        hist: List[Dict[str, Any]] = []
        # warm start: pull each arm once
        for i in range(len(self.arms)):
            s = self.objective_arm(self.arms[i])
            self.counts[i] += 1
            self.values[i] = s
            hist.append({"arm": i, "params": self.arms[i], "score": s})
        for _ in range(len(self.arms), self.budget):
            total = sum(self.counts)
            ucb = [self.values[i] + math.sqrt(2 * math.log(max(total, 1)) / self.counts[i]) for i in range(len(self.arms))]
            i = int(max(range(len(self.arms)), key=lambda j: ucb[j]))
            s = self.objective_arm(self.arms[i])
            self.values[i] = (self.values[i] * self.counts[i] + s) / (self.counts[i] + 1)
            self.counts[i] += 1
            hist.append({"arm": i, "params": self.arms[i], "score": s})
        best_i = int(max(range(len(self.arms)), key=lambda j: self.values[j]))
        return SearchResult(best_params=self.arms[best_i], best_score=self.values[best_i], trials=hist)
