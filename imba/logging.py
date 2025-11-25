from dataclasses import dataclass
from typing import Dict, Any, Optional
import os, json, time
from datetime import datetime

try:
    import mlflow
except Exception:
    mlflow = None


@dataclass
class RunLogger:
    out_dir: str
    use_mlflow: bool = True

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._mlflow_ok = self.use_mlflow and mlflow is not None
        self._start = time.time()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.local_log = os.path.join(self.out_dir, f"run_{self.run_id}.jsonl")
        if self._mlflow_ok:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + os.path.abspath(self.out_dir)))
            mlflow.set_experiment("automl_binary_imbalance")
            self._mlflow = mlflow.start_run(run_name=self.run_id)
        else:
            self._mlflow = None

    def log_params(self, params: Dict[str, Any], step: Optional[int] = None):
        """
        MLflow
        """
        if self._mlflow_ok:
            if step is None:
                to_log = {str(k): str(v) for k, v in params.items()}
            else:
                to_log = {f"t{step}__{k}": str(v) for k, v in params.items()}
            mlflow.log_params(to_log)
        payload = {"type": "params", "params": params}
        if step is not None:
            payload["step"] = step
        self._append(payload)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._mlflow_ok:
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v), step=step if step is not None else 0)
        payload = {"type": "metrics", "metrics": metrics}
        if step is not None:
            payload["step"] = step
        self._append(payload)

    def log_artifact(self, path: str):
        if os.path.exists(path):
            if self._mlflow_ok:
                mlflow.log_artifact(path)
            self._append({"type": "artifact", "path": os.path.abspath(path)})

    def finalize(self):
        duration = time.time() - self._start
        self._append({"type": "done", "secs": duration})
        if self._mlflow_ok:
            mlflow.end_run()

    def _append(self, obj: Dict[str, Any]):
        with open(self.local_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
