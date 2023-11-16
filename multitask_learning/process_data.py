from tasks import (
    TASK_DEFINITION,
    HUGGINGFACE_DATASETS,
    HUGGINGFACE_DATASETS_DEV,
    LOCAL_DATASETS,
    LOCAL_DATASETS_DEV,
    LOCAL_DATA_PATH,
)
import os
from datasets import load_dataset, concatenate_datasets
import tasks
import pandas as pd
import itertools
from tqdm import tqdm
import random

random.seed(42)


def process_trraining_data(max_samples=500000):
    processed_samples = []

    for task in TASK_DEFINITION:
        name = task["name"]
        if name in HUGGINGFACE_DATASETS:
            config = HUGGINGFACE_DATASETS[name]
            data = load_dataset(*config[:-1])[config[-1]]
        elif name in LOCAL_DATASETS:
            path, load_dataset_fn = LOCAL_DATASETS[name]
            if not os.path.isabs(path):
                path = os.path.join(LOCAL_DATA_PATH, path)
            data = load_dataset_fn(path)
        else:
            raise RuntimeError(f"Failed to load {name}")

        if name == "snli":
            data = [sample for sample in data if sample["label"] != -1]

        for sample in tqdm(
            itertools.islice(data, max_samples),
            desc=name,
            total=min(max_samples, len(data)),
        ):
            process_sample_fn = getattr(tasks, f"process_sample_{name}")
            processed_sample = process_sample_fn(sample)
            processed_sample["task"] = name
            processed_samples.append(processed_sample)

    df = pd.DataFrame(processed_samples)
    df = df.sample(frac=1, random_state=42)
    assert len(df[df["label"] < 0]) == 0

    df.to_parquet("train.parquet")


def process_validation_data(max_samples=5000):
    processed_samples = []

    for task in TASK_DEFINITION:
        name = task["name"]
        if name in HUGGINGFACE_DATASETS_DEV:
            config = HUGGINGFACE_DATASETS_DEV[name]
            data = load_dataset(*config[:-1])[config[-1]]
        elif name in LOCAL_DATASETS_DEV:
            path, load_dataset_fn = LOCAL_DATASETS_DEV[name]
            abs_path = os.path.join(LOCAL_DATA_PATH, path)
            data = load_dataset_fn(abs_path)
        else:
            continue

        for sample in tqdm(
            data,
            desc=name,
        ):
            process_sample_fn = getattr(tasks, f"process_sample_{name}")
            processed_sample = process_sample_fn(sample)
            processed_sample["task"] = name
            processed_samples.append(processed_sample)

    df = pd.DataFrame(processed_samples)
    no_label = df["label"] < 0
    assert df[no_label].task.unique().tolist() == ["snli"]
    df = df[~no_label]
    df = df.sample(frac=1, random_state=42)
    df = df.groupby("task").head(max_samples)
    df.to_parquet("validation.parquet")


process_trraining_data()
process_validation_data()
