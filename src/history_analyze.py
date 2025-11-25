import os
import json
import pandas as pd

def analyze_history(history_dir):
    records = []

    for gen in sorted(os.listdir(history_dir), key=lambda x: int(x)):
        gen_path = os.path.join(history_dir, gen)

        for uid in os.listdir(gen_path):
            json_path = os.path.join(gen_path, uid, uid + ".json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            # fitness values (convert from negative to positive)
            metrics = [-v for v in data["fitness"]["wvalues"]]

            # reconstruct pipeline (list operations in order)
            nodes = data["graph"]["operator"]["_nodes"]
            pipeline = " -> ".join([node["content"]["name"] for node in nodes])

            records.append({
                "generation": int(gen),
                "uid": uid,
                "pipeline": pipeline,
                "metric_1": metrics[0],
                "metric_2": metrics[1],
                "time_sec": data["metadata"]["computation_time_in_seconds"]
            })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["generation", "metric_1"], ascending=[True, False])
    return df


# Использование:
df = analyze_history("history1_analcatdata_gsssexsurvey")
pd.set_option('display.max_columns', None)   # показывать все колонки
pd.set_option('display.width', None)        # не ограничивать ширину вывода
pd.set_option('display.max_colwidth', None) # не обрезать длинные значения
print(df)
