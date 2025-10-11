import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(config: dict):
    data = pd.read_csv(config["dataset_path"])
    target_col = config["target_column"]
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )
    return X_train, X_test, y_train, y_test
