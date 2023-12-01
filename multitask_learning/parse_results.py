import json
from collections import defaultdict
from statistics import mean
import pandas as pd

with open("test_logs/lightning_logs/version_8/test_results.json", "r") as f:
    data = json.load(f)

seen, unseen = data["results"]

seen_results = [
    {"name": key.split("/")[0].removeprefix("test_metric_"), "metric_score": value}
    for key, value in seen.items()
]

df_seen = pd.DataFrame(seen_results)

print(df_seen)


results = defaultdict(list)
for key, value in unseen.items():
    dataset, head = key.removeprefix("test_metric_").split(":")
    results[dataset].append((head.split("/")[0], value))

unseen_results = {
    key: {
        "min": min(head_results, key=lambda x: x[1]),
        "max": max(head_results, key=lambda x: x[1]),
        "mean": mean(result[1] for result in head_results),
    }
    for key, head_results in results.items()
}

unseen_results_table = [
    {
        "dataset": key,
        "min": min(result[1] for result in head_results),
        "max": max(result[1] for result in head_results),
        "mean": mean(result[1] for result in head_results),
    }
    for key, head_results in results.items()
]

df_unseen = pd.DataFrame(unseen_results_table)

print(df_unseen)
