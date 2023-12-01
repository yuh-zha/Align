from torch.utils.data import DataLoader, Dataset, ConcatDataset

import pandas as pd
import math
from lightning import LightningDataModule
import os
from datasets import load_dataset
import tasks
from itertools import product
from typing import Optional


class MultitaskTrainDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_parquet(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_dict()
        sample = {k: v for k, v in row.items() if self._valid(v)}
        return sample

    def _valid(self, val):
        return val is not None and (isinstance(val, str) or not math.isnan(val))


_no_op = lambda x: x


class MultitaskDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.train = MultitaskTrainDataset(os.path.join(self.data_dir, "train.parquet"))
        self.validation = MultitaskTrainDataset(
            os.path.join(self.data_dir, "validation.parquet")
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, collate_fn=_no_op, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation, batch_size=self.batch_size, collate_fn=_no_op
        )


class MultitaskTestDataset(Dataset):
    _datasets = dict()

    def __init__(
        self,
        name: str,
        datasset_config,
        *,
        map_fn=None,
        train_name: Optional[str] = None,
        filter_fn=None,
    ):
        self.name = name
        self.train_name = train_name
        self.data = self.get_dataset(datasset_config)
        if filter_fn is not None:
            self.data = self.data.filter(filter_fn)
        train_task = train_name or name
        if map_fn is not None:
            self.map_fn = map_fn
        else:
            self.map_fn = getattr(tasks, f"process_sample_{train_task}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_sample = self.data[idx]
        sample = self.map_fn(raw_sample)
        sample["task"] = self.name
        if self.train_name is not None:
            sample["train_task"] = self.train_name
        return sample

    @classmethod
    def get_dataset(cls, args):
        key = tuple(args)
        if key not in cls._datasets:
            cls._datasets[key] = load_dataset(*args[:-1])[args[-1]]
        return cls._datasets[key]


PAWS_QQP_PATH = "/data/ruichen/General_Alignment_Model/paws_qqp/output/dev_and_test.tsv"


class PawsQQPDataset(Dataset):
    def __init__(self, tsv_path):
        self.data = pd.read_csv(tsv_path, sep="\t")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "task": "paws_qqp",
            "text_a": row.sentence1[2:-1],
            "text_b": row.sentence2[2:-1],
            "label": row.label,
        }


class MultipleChoiceQATestDataset(Dataset):
    def __init__(self, name: str, datasset_config, *, map_fn, train_name=None):
        self.name = name
        self.data = load_dataset(*datasset_config[:-1])[datasset_config[-1]]
        self.map_fn = map_fn
        self.train_name = train_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_sample = self.data[idx]
        context, options, label = self.map_fn(raw_sample)
        sample = {
            "task": self.name,
            "type": "multiple_choice",
            "context": context,
            "options": options,
            "label": label,
        }
        if self.train_name is not None:
            sample["train_task"] = self.train_name
        return sample


def prepare_sciq_sample(sample):
    context = sample["support"]
    option_keys = ("correct_answer", "distractor1", "distractor2", "distractor3")
    question = sample["question"]
    options = [question + " " + sample[key] for key in option_keys]
    return context, options, 0


def _replace_with(text, start, end, replacement):
    return text[:start] + replacement + text[end:]


def prepare_gap_sample(sample):
    text = sample["Text"]
    start = sample["Pronoun-offset"]
    end = start + len(sample["Pronoun"])
    options = [
        _replace_with(text, start, end, sample["A"]),
        _replace_with(text, start, end, sample["B"]),
    ]
    label = int(sample["B-coref"])  # 0 if answer is A, 1 if answer is B
    return text, options, label


def prepare_dream_sample(sample):
    text = " ".join(sample["dialogue"])
    question = sample["question"]
    options = [question + " " + choice for choice in sample["choice"]]
    is_answer = [choice == sample["answer"] for choice in sample["choice"]]
    assert sum(is_answer) == 1
    label = is_answer.index(True)
    return text, options, label


def prepare_quartz_sample(sample):
    text = sample["para"]
    question = sample["question"]
    choices = sample["choices"]
    options = [question + " " + choice for choice in choices["text"]]
    is_answer = [choice == sample["answerKey"] for choice in choices["label"]]
    assert sum(is_answer) == 1
    label = is_answer.index(True)
    return text, options, label


def prepare_quail_sample(sample):
    text = sample["context"]
    question = sample["question"]
    options = [question + " " + choice for choice in sample["answers"]]
    label = sample["correct_answer_id"]
    return text, options, label


def prepare_quail_sample(sample):
    text = sample["context"]
    question = sample["question"]
    options = [question + " " + choice for choice in sample["answers"]]
    label = sample["correct_answer_id"]
    return text, options, label


def _combine_race_qa(question, answer):
    if "_" not in question:
        return question + " " + answer
    return " ".join(question.replace("_", " " + answer + " ").split())


def prepare_race_sample(sample):
    text = sample["article"]
    question = sample["question"]
    options = [_combine_race_qa(question, choice) for choice in sample["options"]]
    label = ord(sample["answer"]) - ord("A")
    return text, options, label


def evaluate_race_m(self):
    true_score = []
    article = []
    qa = []

    for example in self.dataset["race_m"]:
        for i, option in enumerate(example["options"]):
            article.append(example["article"])
            qa.append(
                example["question"] + " " + option + " "
                if "_" not in example["question"]
                else " ".join(
                    example["question"].replace("_", " " + option + " ").split()
                )
            )
            # qa.append(option)
            # true_score.append(i == ord(example['answer'])-65)
            if i == ord(example["answer"]) - 65:
                true_score.append(i)  # 0,1,2,3

    pred_score = []
    pred_score_temp = self.align_func(article, qa)[ALL_TASKS["race_m"]].tolist()
    for a, b, c, d in zip(*[iter(pred_score_temp)] * 4):
        arr = [0] * 4
        # arr[np.argmax([a,b,c,d])] = 1
        # pred_score.extend(arr)
        pred_score.append(np.argmax([a, b, c, d]))

    assert len(pred_score) == len(true_score)
    acc = [int(p == t) for p, t in zip(pred_score, true_score)]
    acc = sum(acc) / len(acc)

    self.print_result_table(
        {
            "Dataset_name": "race-m",
            # 'F1': self.get_f1(true_score, pred_score),
            # 'Accuracy': self.get_accuracy(true_score, pred_score),
            "Accuracy": [acc],
            # 'AUC': roc_auc_score(true_score, pred_score)
        }
    )


def map_to_doc_nli(label_names, premise="premise", hypothesis="hypothesis"):
    doc_nli_label_map = {
        "not_entailment": 0,
        "entailment": 1,
    }
    label_map = {
        i: doc_nli_label_map[label_name] for i, label_name in enumerate(label_names)
    }

    def mapper(sample):
        return {
            "text_a": sample[premise],
            "text_b": sample[hypothesis],
            "label": label_map[sample["label"]],
        }

    return mapper


def map_to_generic_nli(
    label_names, premise="premise", hypothesis="hypothesis", label="label"
):
    nli_label_map = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }
    label_map = {
        i: nli_label_map[label_name] for i, label_name in enumerate(label_names)
    }

    def mapper(sample):
        return {
            "text_a": sample[premise],
            "text_b": sample[hypothesis],
            "label": label_map[sample[label]],
        }

    return mapper


def map_binary_nli_with_label_map(
    label_names, premise="premise", hypothesis="hypothesis", label="label"
):
    # Treat 3 way NLI neutral & contradiction as not_entailment
    acceptable_label_map = {
        "entailment": (0,),  # 3 way NLI entailment
        "not_entailment": (1, 2),  # 3 way NLI neutral & contradiction
    }
    label_map = [acceptable_label_map[label_name] for label_name in label_names]

    def mapper(sample):
        return {
            "text_a": sample[premise],
            "text_b": sample[hypothesis],
            "label": sample[label],
            "type": "classification_with_label_map",
            "label_map": label_map,
        }

    return mapper


def get_test_dataset(paws_qqp_tsv_path=PAWS_QQP_PATH):
    return ConcatDataset(
        (
            MultitaskTestDataset(
                "mnli_matched",
                ["multi_nli", "validation_matched"],
                train_name="mnli",
            ),
            MultitaskTestDataset(
                "mnli_mismatched",
                ["multi_nli", "validation_mismatched"],
                train_name="mnli",
            ),
            MultitaskTestDataset("anli_r1", ["anli", "test_r1"]),
            MultitaskTestDataset("anli_r2", ["anli", "test_r2"]),
            MultitaskTestDataset("anli_r3", ["anli", "test_r3"]),
            MultitaskTestDataset(
                "snli",
                ["snli", "test"],
                filter_fn=lambda sample: sample["label"] != -1,
            ),
            MultitaskTestDataset("nli_fever", ["pietrolesci/nli_fever", "dev"]),
            MultitaskTestDataset("vitaminc", ["tals/vitaminc", "test"]),
            MultitaskTestDataset("sick", ["sick", "default", "test"]),
            MultitaskTestDataset("stsb", ["glue", "stsb", "validation"]),
            MultitaskTestDataset("paws", ["paws", "labeled_final", "test"]),
            PawsQQPDataset(paws_qqp_tsv_path),
            MultitaskTestDataset("qqp", ["glue", "qqp", "validation"]),
            MultitaskTestDataset("eraser_multi_rc", ["eraser_multi_rc", "test"]),
            MultitaskTestDataset("boolq", ["boolq", "validation"]),
            MultipleChoiceQATestDataset(
                "sciq",
                ["sciq", "test"],
                map_fn=prepare_sciq_sample,
            ),
            MultipleChoiceQATestDataset(
                "gap",
                ["gap", "test"],
                map_fn=prepare_gap_sample,
            ),
            MultipleChoiceQATestDataset(
                "race_m",
                ["race", "middle", "test"],
                train_name="race",
                map_fn=prepare_race_sample,
            ),
            MultipleChoiceQATestDataset(
                "race_h",
                ["race", "high", "test"],
                train_name="race",
                map_fn=prepare_race_sample,
            ),
            MultipleChoiceQATestDataset(
                "quail",
                ["quail", "challenge"],
                map_fn=prepare_quail_sample,
            ),
        )
    )


def get_unseen_test_dataset():
    combinations = [
        (  # paraphrase detection
            ("paws", "paws_qqp", "qqp"),
            (
                lambda train_task: MultitaskTestDataset(
                    f"mrpc:{train_task}",
                    ["glue", "mrpc", "test"],
                    train_name=train_task,
                    map_fn=tasks.process_sample_paws,
                ),
            ),
        ),
        (  # QA
            (
                "race",
                "eraser_multi_rc",
                "drop",
                "hotpot_qa_distractor",
                "hotpot_qa_fullwiki",
                "quail",
                "sciq",
            ),
            (
                lambda train_task: MultipleChoiceQATestDataset(
                    f"dream:{train_task}",
                    ["dream", "test"],
                    train_name=train_task,
                    map_fn=prepare_dream_sample,
                ),
                lambda train_task: MultipleChoiceQATestDataset(
                    f"quartz:{train_task}",
                    ["quartz", "test"],
                    train_name=train_task,
                    map_fn=prepare_quartz_sample,
                ),
            ),
        ),
        (  # binary NLI
            ("doc_nli",),
            (
                lambda train_task: MultitaskTestDataset(
                    f"axb:{train_task}",
                    ["super_glue", "axb", "test"],
                    train_name=train_task,
                    map_fn=map_to_doc_nli(
                        ("entailment", "not_entailment"),
                        premise="sentence1",
                        hypothesis="sentence2",
                    ),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"axg:{train_task}",
                    ["super_glue", "axg", "test"],
                    train_name=train_task,
                    map_fn=map_to_doc_nli(("entailment", "not_entailment")),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"rte:{train_task}",
                    ["super_glue", "rte", "validation"],
                    train_name=train_task,
                    map_fn=map_to_doc_nli(("entailment", "not_entailment")),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"wnli:{train_task}",
                    ["SetFit/wnli", "validation"],
                    train_name=train_task,
                    map_fn=map_to_doc_nli(
                        ("not_entailment", "entailment"),
                        premise="text1",
                        hypothesis="text2",
                    ),
                ),
            ),
        ),
        (  # generic 3 way NLI
            ("snli", "mnli", "anli_r1", "anli_r2", "anli_r3"),
            (
                lambda train_task: MultitaskTestDataset(
                    f"cb:{train_task}",
                    ["super_glue", "cb", "validation"],
                    train_name=train_task,
                    map_fn=map_to_generic_nli(
                        ("entailment", "contradiction", "neutral")
                    ),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"se14t1:{train_task}",
                    ["sem_eval_2014_task_1", "test"],
                    train_name=train_task,
                    map_fn=map_to_generic_nli(
                        ("neutral", "entailment", "contradiction"),
                        label="entailment_judgment",
                    ),
                ),
                # Also try using 3 way NLI head for these binary NLI datasets
                # by mapping neutral & contradiction to not_entailment.
                lambda train_task: MultitaskTestDataset(
                    f"axb:{train_task}",
                    ["super_glue", "axb", "test"],
                    train_name=train_task,
                    map_fn=map_binary_nli_with_label_map(
                        ("entailment", "not_entailment"),
                        premise="sentence1",
                        hypothesis="sentence2",
                    ),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"axg:{train_task}",
                    ["super_glue", "axg", "test"],
                    train_name=train_task,
                    map_fn=map_binary_nli_with_label_map(
                        ("entailment", "not_entailment")
                    ),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"rte:{train_task}",
                    ["super_glue", "rte", "validation"],
                    train_name=train_task,
                    map_fn=map_binary_nli_with_label_map(
                        ("entailment", "not_entailment")
                    ),
                ),
                lambda train_task: MultitaskTestDataset(
                    f"wnli:{train_task}",
                    ["SetFit/wnli", "validation"],
                    train_name=train_task,
                    map_fn=map_binary_nli_with_label_map(
                        ("not_entailment", "entailment"),
                        premise="text1",
                        hypothesis="text2",
                    ),
                ),
            ),
        ),
    ]
    datasets = []
    for train_tasks, get_dataset_fns in combinations:
        for train_task, get_dataset_fn in product(train_tasks, get_dataset_fns):
            datasets.append(get_dataset_fn(train_task))
    return ConcatDataset(datasets)
