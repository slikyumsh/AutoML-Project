import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, 
                           precision_score, recall_score, average_precision_score,
                           classification_report, confusion_matrix, log_loss)
from datetime import datetime

def evaluate_model(model, X_test, y_test, config: dict, train_info: dict = None):
    test_data = pd.concat([X_test, y_test.rename(config["target_column"])], axis=1)
    pred_proba = model.predict(test_data).data[:, 0]
    pred_binary = (pred_proba > 0.5).astype(int)
    
    metrics = {}
    
    # Основные метрики
    metrics["ROC_AUC"] = roc_auc_score(y_test, pred_proba)
    metrics["PR_AUC"] = average_precision_score(y_test, pred_proba)
    metrics["Accuracy"] = accuracy_score(y_test, pred_binary)
    metrics["Precision"] = precision_score(y_test, pred_binary, zero_division=0)
    metrics["Recall"] = recall_score(y_test, pred_binary, zero_division=0)
    metrics["F1"] = f1_score(y_test, pred_binary, zero_division=0)
    metrics["Log_Loss"] = log_loss(y_test, pred_proba)
    
    # Дополнительные метрики
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["False_Positive_Rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["Balanced_Accuracy"] = (metrics["Recall"] + metrics["Specificity"]) / 2
    
    # Добавляем информацию об обучении
    if train_info:
        metrics.update(train_info)
    
    save_comprehensive_report(metrics, config["metrics_path"], y_test, pred_binary, pred_proba)
    return metrics

def save_comprehensive_report(metrics: dict, path: str, y_true, y_pred, y_proba):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("🚀 LightAutoML Comprehensive Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Информация о данных
        f.write("📊 DATA INFORMATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Train samples: {metrics.get('train_samples', 'N/A')}\n")
        f.write(f"Test samples: {metrics.get('test_samples', 'N/A')}\n")
        f.write(f"Class balance (train): {metrics.get('class_balance', 'N/A')}\n")
        f.write(f"Features: {metrics.get('n_features', 'N/A')}\n\n")
        
        # Информация о тренировке
        f.write("⏰ TRAINING INFORMATION\n")
        f.write("-" * 25 + "\n")
        f.write(f"Allocated time: {metrics.get('allocated_time', 'N/A')} sec\n")
        f.write(f"Actual training time: {metrics.get('actual_training_time', 'N/A'):.1f} sec\n")
        f.write(f"Models built: {metrics.get('models_built', 'N/A')}\n\n")
        
        # Метрики
        f.write("📈 MODEL METRICS\n")
        f.write("-" * 15 + "\n")
        for name, value in metrics.items():
            if name not in ['train_samples', 'test_samples', 'class_balance', 'n_features', 
                           'allocated_time', 'actual_training_time', 'models_built', 
                           'model_types', 'feature_importance']:
                if isinstance(value, (int, float)):
                    f.write(f"{name}: {value:.4f}\n")
                else: 
                    f.write(f"{name}: {value}\n")
        # Информация о моделях
        f.write(f"\n🤖 MODELS BUILT ({metrics.get('models_built', 0)})\n")
        f.write("-" * 25 + "\n")
        for model_info in metrics.get('model_types', []):
            f.write(f"- {model_info}\n")
        
        # Важность признаков
        if 'feature_importance' in metrics:
            f.write(f"\n🎯 TOP 10 FEATURE IMPORTANCE\n")
            f.write("-" * 30 + "\n")
            print(metrics['feature_importance'])
            for feature, importance in metrics['feature_importance'][:10]:
                f.write(f"{feature}: {importance:.4f}\n")
        
        # Confusion Matrix
        f.write(f"\n🔢 CONFUSION MATRIX\n")
        f.write("-" * 20 + "\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"True Negatives: {cm[0,0]}\n")
        f.write(f"False Positives: {cm[0,1]}\n") 
        f.write(f"False Negatives: {cm[1,0]}\n")
        f.write(f"True Positives: {cm[1,1]}\n")
        
        f.write(f"\n📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")