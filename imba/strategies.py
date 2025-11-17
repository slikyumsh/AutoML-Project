# Two search strategies: Bayesian (hyperopt TPE) and Multi-Armed Bandit (UCB1)
from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import math
import numpy as np

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
except Exception:
    fmin = None  # handled in BayesianSearch.__init__


@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]]


# --------------------
# Bayesian via hyperopt
# --------------------
class BayesianSearch:
    def __init__(
        self,
        space: Dict[str, Any],
        objective: Callable[[Dict[str, Any]], float],
        max_evals: int = 50,
        seed: int = 42,
    ):
        if fmin is None:
            raise ImportError("hyperopt is required for BayesianSearch")
        self.space = space
        self.objective = objective
        self.max_evals = max_evals
        self.seed = seed

    def run(self) -> SearchResult:
        trials = Trials()

        def _obj(params):
            score = self.objective(params)
            return {"loss": -score, "status": STATUS_OK}

        rng = np.random.default_rng(self.seed)  # numpy Generator required by hyperopt
        best = fmin(
            fn=_obj,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=rng,
        )

        hist: List[Dict[str, Any]] = []
        for t in trials.trials:
            p = t["misc"]["vals"]
            flat = {k: (v[0] if isinstance(v, list) and v else v) for k, v in p.items()}
            hist.append({"params": flat, "score": -t["result"]["loss"]})

        best_score = max((h["score"] for h in hist), default=float("nan"))
        return SearchResult(best_params=best, best_score=best_score, trials=hist)


# --------------------
# Multi-armed bandit (UCB1) over discrete arms
# --------------------
class BanditUCB1:
    def __init__(
        self,
        arms: List[Dict[str, Any]],
        objective_arm: Callable[[Dict[str, Any]], float],
        budget: int = 60,
    ):
        self.arms = arms
        self.objective_arm = objective_arm
        self.budget = budget
        self.counts = [0] * len(arms)
        self.values = [0.0] * len(arms)

    def run(self) -> SearchResult:
        history: List[Dict[str, Any]] = []

        # Pull each arm once for initialization
        for i in range(len(self.arms)):
            score = self.objective_arm(self.arms[i])
            self.counts[i] += 1
            self.values[i] = score
            history.append({"arm": i, "params": self.arms[i], "score": score})

        # Main loop
        for _ in range(len(self.arms), self.budget):
            total = sum(self.counts)
            ucb = [
                self.values[i] + math.sqrt(2 * math.log(max(total, 1)) / self.counts[i])
                for i in range(len(self.arms))
            ]
            i = int(max(range(len(self.arms)), key=lambda j: ucb[j]))
            score = self.objective_arm(self.arms[i])
            self.values[i] = (self.values[i] * self.counts[i] + score) / (self.counts[i] + 1)
            self.counts[i] += 1
            history.append({"arm": i, "params": self.arms[i], "score": score})

        best_idx = int(max(range(len(self.arms)), key=lambda j: self.values[j]))
        return SearchResult(best_params=self.arms[best_idx], best_score=self.values[best_idx], trials=history)
