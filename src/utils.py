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

def get_data_stats(X_train, X_test, y_train, y_test) -> dict:
    """Собирает статистику о данных"""
    return {
        'train_samples': len(X_train),
        'test_samples': len(X_test), 
        'n_features': X_train.shape[1],
        'class_balance_train': f"{y_train.mean():.3f} / {1 - y_train.mean():.3f}",
        'class_balance_test': f"{y_test.mean():.3f} / {1 - y_test.mean():.3f}"
    }

def load_data_with_mapping(config: dict):
    """Загрузка данных с явным указанием mapping для целевой переменной"""
    data = pd.read_csv(config["dataset_path"])
    target_col = config["target_column"]
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    print(f"🎯 Target variable info:")
    print(f"   dtype: {y.dtype}")
    print(f"   unique values: {sorted(y.unique())}")
    
    # Применяем mapping ТОЛЬКО если целевая переменная - строки
    if config.get("target_mapping") and y.dtype == 'object':
        mapping = config["target_mapping"]
        print(f"   Applying mapping: {mapping}")
        y = y.map(mapping)
        
        # Проверяем на NaN после mapping
        if y.isna().any():
            nan_count = y.isna().sum()
            print(f"⚠️ Warning: {nan_count} NaN values after mapping")
            print("   Removing rows with NaN in target...")
            
            # Удаляем строки с NaN
            nan_mask = y.isna()
            X = X[~nan_mask]
            y = y[~nan_mask]
            print(f"   Remaining samples: {len(X)}")
    
    # Если уже числа - оставляем как есть
    elif pd.api.types.is_numeric_dtype(y):
        y = y - 1

        # Обработка булевых значений
    elif y.dtype == 'bool':
        print("   🔧 Converting boolean to numeric (True=1, False=0)")
        y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )
    
    # print(f"✅ Final split: {len(X_train)} train, {len(X_test)} test")
    # print(f"✅ Target values in train: {sorted(y_train.unique())}")
    
    return X_train, X_test, y_train, y_test