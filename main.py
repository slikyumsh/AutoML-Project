from src.utils import load_config, load_data, get_data_stats, load_data_with_mapping
from src.train import train_model
from src.eval import evaluate_model

def main():
    config = load_config("config.json")
    X_train, X_test, y_train, y_test = load_data_with_mapping(config)
    
    # Собираем статистику данных
    data_stats = get_data_stats(X_train, X_test, y_train, y_test)
    
    # Обучаем модель и получаем информацию о тренировке
    model, _, train_info = train_model(X_train, y_train, config)
    
    # Объединяем всю информацию
    full_info = {**data_stats, **train_info}
    # Оцениваем модель с полной информацией
    metrics = evaluate_model(model, X_test, y_test, config, full_info)
    
    print("\n✅ Comprehensive report saved to:", config["metrics_path"])
    print("\n📊 Key Metrics:")
    for k, v in metrics.items():
        if k in ['ROC_AUC', 'PR_AUC', 'F1', 'Accuracy']:
            print(f"   {k}: {v:.4f}")

if __name__ == "__main__":
    main()