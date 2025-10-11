from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import joblib
import pandas as pd


def train_model(X_train, y_train, config: dict):
    task = Task(config["task"])
    
    # объединяем признаки и таргет в один DataFrame
    train_data = pd.concat([X_train, y_train.rename(config["target_column"])], axis=1)
    
    automl = TabularAutoML(
        task=task,
        timeout=config["automl_timeout"],
        cpu_limit=config["cpu_limit"],
        general_params={"use_algos": config["use_algos"]}
    )
    
    oof_pred = automl.fit_predict(train_data, roles={"target": config["target_column"]})
    
    joblib.dump(automl, config["output_model_path"])
    return automl, oof_pred
