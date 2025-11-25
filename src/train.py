from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco
import joblib
import pandas as pd
import time
from typing import Tuple, Dict, Any
import os

def train_model(X_train, y_train, config: dict) -> Tuple[Any, Any, Dict]:
    start_time = time.time()
    
    task = Task(config["task"])
    
    # Анализ данных перед обучением
    train_info = {
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'class_balance': f"{y_train.mean():.3f} / {1 - y_train.mean():.3f}",
        'allocated_time': config["automl_timeout"]
    }
    
    # объединяем признаки и таргет в один DataFrame
    train_data = pd.concat([X_train, y_train.rename(config["target_column"])], axis=1)
    
    automl = TabularAutoML(
        task=task,
        timeout=config["automl_timeout"],
        cpu_limit=config["cpu_limit"],
        general_params={"use_algos": config["use_algos"]}
    )
    
    RD = ReportDeco(output_path="report")
    automl_rd = RD(automl)
    oof_pred = automl_rd.fit_predict(train_data, roles={"target": config["target_column"]})
    
    # Время обучения
    train_info['actual_training_time'] = time.time() - start_time
    
    # Информация о моделях
    train_info.update(extract_model_info(automl_rd))
    
    # Важность признаков
    train_info['feature_importance'] = extract_feature_importance(automl_rd, X_train.columns)

    os.makedirs(os.path.dirname(config["output_model_path"]), exist_ok=True)
    joblib.dump(automl_rd, config["output_model_path"])
    return automl_rd, oof_pred, train_info

def extract_model_info(automl) -> Dict[str, Any]:
    """Извлекает информацию о построенных моделях"""
    
    # Получаем настоящую модель из-под ReportDeco
    if hasattr(automl, '_model'):
        real_automl = automl._model
    else:
        real_automl = automl
    
    model_info = {
        'models_built': 0,
        'model_types': []
    }
    
    try:
        # Проходим по всем уровням и моделям
        for level_idx, level in enumerate(real_automl.levels):
            for model in level:
                if hasattr(model, 'ml_algos'):
                    # Это пайплайн с несколькими моделями
                    for ml_algo in model.ml_algos:
                        model_info['models_built'] += 1
                        
                        if hasattr(ml_algo, 'name'):
                            model_name = ml_algo.name
                        else:
                            model_name = type(ml_algo).__name__
                        
                        clean_name = model_name.replace('Lvl_', '').replace('Pipe_', '').replace('Mod_', '').replace('Tuned_', 'Tuned ')
                        model_type = type(ml_algo).__name__
                        
                        # Добавляем информацию о тюнинге
                        tuned_info = " (tuned)" if "tuned" in model_name.lower() else ""
                        
                        model_info['model_types'].append(f"L{level_idx+1}: {clean_name}{tuned_info}")
                
    except Exception as e:
        model_info['model_types'].append(f"Error extracting model info: {str(e)}")
    
    return model_info


def extract_feature_importance(automl, feature_names) -> list:
    """Извлекает важность признаков"""
    try:
        importance_df = automl.model.get_feature_scores("fast")
        
        importance_list = []
        for _, row in importance_df.iterrows():
            importance_list.append((row['Feature'], row['Importance']))
        
        importance_list.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🎯 Top 5 features: {importance_list[:5]}")
        return importance_list
        
    except Exception as e:
        print(f"⚠️ Error extracting feature importance: {e}")
    
    # Если не получилось, возвращаем заглушку
    return [(name, 0.0) for name in feature_names]