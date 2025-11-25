import openml
import pandas as pd
from fedot import FedotBuilder
from fedot.core.pipelines.node import SecondaryNode
from sklearn.model_selection import train_test_split
from openml import datasets
from sklearn.preprocessing import LabelEncoder
import traceback
from fedot.core.pipelines.pipeline import Pipeline
import numpy as np


# === 1. Загружаем CSV с датасетами ===
meta = pd.read_csv('csv_folder/bucket_le_50000_90_10.csv')

results = []


# Функция для получения важности признаков из финального пайплайна
def get_feature_importances(fedot_pipeline: Pipeline, X_data: pd.DataFrame, n_top: int = 5):
    """Извлекает важность признаков из первого основного классификатора в пайплайне."""

    # 1. Поиск первого узла-классификатора, который поддерживает feature_importances_
    # FEDOT использует узлы-обертки, поэтому нужно искать в них.
    classifier_node = None
    for node in reversed(fedot_pipeline.nodes):
        if isinstance(node, SecondaryNode) and node.content['name'] in ['rf', 'xgboost', 'lgbm', 'catboost']:
            classifier_node = node
            break

    if not classifier_node:
        return {'top_features': 'No tree-based classifier found'}

    try:
        # Модель хранится во внутреннем объекте operation.fitted_operation
        model = classifier_node.operation.fitted_operation

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return {
                'top_features': f'Classifier {classifier_node.content["name"]} does not support feature_importances_'}

        # 2. Сортировка и выбор топ N
        feature_names = X_data.columns if isinstance(X_data, pd.DataFrame) else [f'feature_{i}' for i in
                                                                                 range(X_data.shape[1])]

        if len(importances) != len(feature_names):
            # Эта проблема часто возникает из-за препроцессоров (PCA, PolyFeatures и т.д.)
            return {'top_features': f'Cannot map importances due to complex preproc (len={len(importances)})'}

        sorted_indices = np.argsort(importances)[::-1]
        top_features = {
            feature_names[i]: float(importances[i])
            for i in sorted_indices[:n_top]
        }
        return {'top_features': top_features}

    except Exception as e:
        return {'top_features': f'Error extracting importances: {e}'}

# === 2. Проходим по каждому датасету ===
for idx, row in meta.iterrows():
    did = row['did']
    name = row['name']

    print(f'\n==============================')
    print(f'Processing dataset: {name} (did={did})')
    print(f'==============================\n')

    try:
        # === 3. Загружаем датасет из OpenML ===
        dataset = datasets.get_dataset(did)
        X, y, categorical, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        # Если целевой нет – пропуск
        if y is None:
            print(f'❌ No target column in dataset {name}, skipping')
            continue

        # === 4. Кодируем целевую переменную (если строки) ===
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # === 5. Делим на train/test ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # === 6. Создаём и обучаем FEDOT ===
        fedot = (FedotBuilder(problem='classification')
                 .setup_output(history_dir=f"history1_{name}")
                 .setup_composition(timeout=10,
                                    with_tuning=True,
                                    preset='best_quality')
                 .setup_pipeline_evaluation(max_pipeline_fit_time=5,
                                            metric=['roc_auc', 'f1'])
                 .build())

        fedot.fit(features=X_train, target=y_train)

        # === 7. Получаем метрики ===
        _ = fedot.predict(features=X_test)
        metrics = fedot.get_metrics(y_test)
        print(f"✔ Metrics for {name}: {metrics}")

        feature_importance_result = get_feature_importances(fedot.current_pipeline, X_train, n_top=3)
        print(f"⭐ Top Features for {name}: {feature_importance_result['top_features']}")

        # === 8. Сохраняем результат ===
        results.append({
            'did': did,
            'name': name,
            'roc_auc': metrics.get('roc_auc'),
            'f1': metrics.get('f1'),
            'top_features': str(feature_importance_result['top_features'])
        })

    except Exception as e:
        print(f'❌ Error on dataset {name}: {e}')
        traceback.print_exc()

        results.append({
            'did': did,
            'name': name,
            'roc_auc': None,
            'f1': None,
            'error': str(e)
        })
        continue

# === 9. Сохраняем таблицу результатов ===
results_df = pd.DataFrame(results)
results_df.to_csv('fedot_results_bucket_le_50000_90_10.csv', index=False)

print("\n\n==============================")
print("DONE! Results saved to fedot_results.csv")
print("==============================")