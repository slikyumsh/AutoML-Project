from src.utils import load_config, load_data
from src.train import train_model
from src.eval import evaluate_model


def main():
    config = load_config("config.json")
    X_train, X_test, y_train, y_test = load_data(config)
    
    model, _ = train_model(X_train, y_train, config)
    metrics = evaluate_model(model, X_test, y_test, config)
    
    print("✅ Evaluation metrics saved to:", config["metrics_path"])
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
