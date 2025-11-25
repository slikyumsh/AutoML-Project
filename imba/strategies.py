from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
import optuna


@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]]


class OptunaSearch:
    """Wrapper over Optuna Study; objective(trial)->float maximize."""
    def __init__(
        self,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 50,
        seed: int = 42,
        study_name: str = "automl_optuna",
        direction: str = "maximize",
    ):
        self.objective = objective
        self.n_trials = n_trials
        self.seed = seed
        self.study_name = study_name
        self.direction = direction
        self.study: Optional[optuna.Study] = None

    def run(self) -> SearchResult:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler
        )
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)

        hist = []
        for t in self.study.trials:
            if t.value is None:
                continue
            hist.append({"number": t.number, "params": t.params, "score": t.value})

        best = self.study.best_trial
        return SearchResult(best_params=best.params, best_score=float(best.value), trials=hist)
