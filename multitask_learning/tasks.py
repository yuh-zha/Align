import json
from tqdm import tqdm
import jsonlines
import random

TASK_DEFINITION = (
    {"name": "snli", "type": "classification", "num_labels": 3},
    {"name": "mnli", "type": "classification", "num_labels": 3},
    {"name": "anli_r1", "type": "classification", "num_labels": 3},
    {"name": "anli_r2", "type": "classification", "num_labels": 3},
    {"name": "anli_r3", "type": "classification", "num_labels": 3},
    {"name": "doc_nli", "type": "classification", "num_labels": 2},
    {"name": "nli_fever", "type": "classification", "num_labels": 3},
    {"name": "vitaminc", "type": "classification", "num_labels": 3},
    {"name": "qqp", "type": "classification", "num_labels": 2},
    {"name": "paws", "type": "classification", "num_labels": 2},
    {"name": "paws_qqp", "type": "classification", "num_labels": 2},
    {"name": "wiki103", "type": "classification", "num_labels": 2},
    {"name": "sick", "type": "regression"},
    {"name": "stsb", "type": "regression"},
    {"name": "squad_v2", "type": "reading_comprehension"},
    {"name": "race", "type": "classification", "num_labels": 2},
    {"name": "adversarial_qa", "type": "reading_comprehension"},
    {"name": "boolq", "type": "classification", "num_labels": 2},
    {"name": "drop", "type": "classification", "num_labels": 2},
    {"name": "eraser_multi_rc", "type": "classification", "num_labels": 2},
    {"name": "hotpot_qa_distractor", "type": "classification", "num_labels": 2},
    {"name": "hotpot_qa_fullwiki", "type": "classification", "num_labels": 2},
    {"name": "newsqa", "type": "reading_comprehension"},
    {"name": "quail", "type": "classification", "num_labels": 2},
    {"name": "quoref", "type": "reading_comprehension"},
    {"name": "ropes", "type": "classification", "num_labels": 2},
    {"name": "sciq", "type": "classification", "num_labels": 2},
    {"name": "strategy_qa", "type": "classification", "num_labels": 2},
    {"name": "msmarco", "type": "classification", "num_labels": 2},
    {"name": "wikihow", "type": "classification", "num_labels": 2},
    {"name": "gap", "type": "classification", "num_labels": 2},
)

HUGGINGFACE_DATASETS = {
    "mnli": ["multi_nli", "default", "train"],
    "squad_v2": ["squad_v2", "squad_v2", "train"],
    "paws": ["paws", "labeled_final", "train"],
    "vitaminc": ["tals/vitaminc", "v1.0", "train"],
    "stsb": ["glue", "stsb", "train"],
    "sick": ["sick", "default", "train"],
    "anli_r1": ["anli", "plain_text", "train_r1"],
    "anli_r2": ["anli", "plain_text", "train_r2"],
    "anli_r3": ["anli", "plain_text", "train_r3"],
    "snli": ["snli", "plain_text", "train"],
    "qqp": ["glue", "qqp", "train"],
    "adversarial_qa": ["adversarial_qa", "adversarialQA", "train"],
    "quoref": ["quoref", "train"],
    "boolq": ["boolq", "train"],
    "eraser_multi_rc": ["eraser_multi_rc", "train"],
    "strategy_qa": ["metaeval/strategy-qa", "train"],
    "doc_nli": ["saattrupdan/doc-nli", "train"],
    "nli_fever": ["pietrolesci/nli_fever", "train"],
    "vitaminc": ["tals/vitaminc", "train"],
}

HUGGINGFACE_DATASETS_DEV = {
    "paws": ["paws", "labeled_final", "validation"],
    "vitaminc": ["tals/vitaminc", "v1.0", "validation"],
    "sick": ["sick", "default", "validation"],
    "anli_r1": ["anli", "plain_text", "dev_r1"],
    "anli_r2": ["anli", "plain_text", "dev_r2"],
    "anli_r3": ["anli", "plain_text", "dev_r3"],
    "snli": ["snli", "plain_text", "validation"],
    "adversarial_qa": ["adversarial_qa", "adversarialQA", "validation"],
    "quoref": ["quoref", "validation"],
    "eraser_multi_rc": ["eraser_multi_rc", "validation"],
    "doc_nli": ["saattrupdan/doc-nli", "val"],
    "vitaminc": ["tals/vitaminc", "validation"],
}


def load_newsqa(split):
    def do_load_newsqa(path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        samples = []
        for story in tqdm(data["data"], desc="Loading newsqa"):
            if story["type"] != split:
                continue
            context = story["text"]
            for question in story["questions"]:
                if question.get("isQuestionBad", 0.0) > 0.2:
                    continue
                if "s" in question["consensus"]:
                    start = question["consensus"]["s"]
                    end = question["consensus"]["e"]
                    label = (start, end, True)
                else:
                    label = (-100, -100, False)
                label_start, label_end, label_ans = label
                samples.append(
                    {
                        "question": question["q"],
                        "context": context,
                        "label_start": label_start,
                        "label_end": label_end,
                        "label_ans": label_ans,
                    }
                )
        return samples

    return do_load_newsqa


def load_align_dataset(path):
    with jsonlines.open(path, "r") as reader:
        data = list(reader)
    return data


# def load_wikihow(path):
#     data = []
#     count = []
#     with jsonlines.open(path, "r") as reader:
#         for sample in reader:
#             count.append(len(sample["text_b"]) + len(sample["text_c"]))
#             assert sample["orig_label"] == -1
#             text_a = sample["text_a"]
#             for text_b in sample["text_b"]:
#                 data.append(
#                     {
#                         "text_a": text_a,
#                         "text_b": [text_b],
#                         "orig_label": 1,
#                     }
#                 )
#             for text_c in sample["text_c"]:
#                 data.append(
#                     {
#                         "text_a": text_a,
#                         "text_b": [text_c],
#                         "orig_label": 0,
#                     }
#                 )
#     # this is effectively what we did by samples one text_b/text_c per sample
#     samples = random.sample(data, k=len(count))
#     return samples


LOCAL_DATA_PATH = "/home/yuheng/DST_Alignment_Model/data/training"

LOCAL_DATASETS = {
    "paws_qqp": ("paws_qqp.json", load_align_dataset),
    "wiki103": ("wiki103.json", load_align_dataset),
    "race": ("race.json", load_align_dataset),
    "drop": ("drop.json", load_align_dataset),
    "hotpot_qa_distractor": ("hotpot_qa_distractor.json", load_align_dataset),
    "hotpot_qa_fullwiki": ("hotpot_qa_fullwiki.json", load_align_dataset),
    "quail": ("quail.json", load_align_dataset),
    "ropes": ("ropes.json", load_align_dataset),
    "sciq": ("sciq.json", load_align_dataset),
    "msmarco": ("msmarco.json", load_align_dataset),
    "wikihow": ("wikihow.json", load_align_dataset),
    "gap": ("gap.json", load_align_dataset),
    "newsqa": (
        "/home/yichi/multitask/newsqa_raw/combined-newsqa-data-v1.json",
        load_newsqa("train"),
    ),
}

LOCAL_DATASETS_DEV = {
    "newsqa": (
        "/home/yichi/multitask/newsqa_raw/combined-newsqa-data-v1.json",
        load_newsqa("dev"),
    ),
}


def _generic_nli(sample):
    return {
        "text_a": sample["premise"],
        "text_b": sample["hypothesis"],
        "label": sample["label"],
    }


process_sample_snli = _generic_nli
process_sample_mnli = _generic_nli
process_sample_anli_r1 = _generic_nli
process_sample_anli_r2 = _generic_nli
process_sample_anli_r3 = _generic_nli


def process_sample_doc_nli(sample):
    label_map = {
        "not_entailment": 0,
        "entailment": 1,
    }
    return {
        "text_a": sample["premise"],
        "text_b": sample["hypothesis"],
        "label": label_map[sample["label"]],
    }


process_sample_nli_fever = _generic_nli


def process_sample_vitaminc(sample):
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    return {
        "text_a": sample["evidence"],
        "text_b": sample["claim"],
        "label": label_map[sample["label"]],
    }


def process_sample_qqp(sample):
    return {
        "text_a": sample["question1"],
        "text_b": sample["question2"],
        "label": sample["label"],
    }


def process_sample_paws(sample):
    return {
        "text_a": sample["sentence1"],
        "text_b": sample["sentence2"],
        "label": sample["label"],
    }


def _align_paraphrase(sample):
    assert len(sample["text_b"]) == 1
    assert not sample["text_c"]
    return {
        "text_a": sample["text_a"],
        "text_b": sample["text_b"][0],
        "label": sample["orig_label"],
    }


def _align_qa(sample):
    assert len(sample["text_b"]) == 1
    if sample["text_c"]:
        text_b = sample["text_b"][0] + " " + sample["text_c"][0]
    else:
        text_b = sample["text_b"][0]
    return {
        "text_a": sample["text_a"],
        "text_b": text_b,
        "label": sample["orig_label"],
    }


def _align_summarization(sample):
    text_a = sample["text_a"]
    if random.random() > 0.5:  # this will be a positive pair
        random_pos_sample_id = random.randint(0, len(sample["text_b"]) - 1)
        text_b = sample["text_b"][random_pos_sample_id]
        label = 1
    else:  # this will be a negative pair
        assert sample["text_c"]
        random_neg_sample_id = random.randint(0, len(sample["text_c"]) - 1)
        text_b = sample["text_c"][random_neg_sample_id]
        label = 0
    return {
        "text_a": text_a,
        "text_b": text_b,
        "label": label,
    }


process_sample_paws_qqp = _align_paraphrase
process_sample_wiki103 = _align_paraphrase


def process_sample_sick(sample):
    return {
        "text_a": sample["sentence_A"],
        "text_b": sample["sentence_B"],
        "label": sample["relatedness_score"] / 5,
    }


def process_sample_stsb(sample):
    return {
        "text_a": sample["sentence1"],
        "text_b": sample["sentence2"],
        "label": sample["label"] / 5,
    }


def process_sample_squad_v2(sample):
    answers = list(zip(sample["answers"]["text"], sample["answers"]["answer_start"]))
    if sample["answers"]["text"]:
        text, start = answers[0]
        label = (start, start + len(text), True)
    else:
        label = (-100, -100, False)
    label_start, label_end, label_ans = label
    return {
        "question": sample["question"],
        "context": sample["context"],
        "label_start": label_start,
        "label_end": label_end,
        "label_ans": label_ans,
    }


process_sample_race = _align_qa

process_sample_adversarial_qa = process_sample_squad_v2


def process_sample_boolq(sample):
    return {
        "text_a": sample["question"],
        "text_b": sample["passage"],
        "label": int(sample["answer"]),
    }


process_sample_drop = _align_qa


def process_sample_eraser_multi_rc(sample):
    return {
        "text_a": sample["passage"],
        "text_b": sample["query_and_answer"].replace("|", ""),
        "label": sample["label"],
    }


process_sample_hotpot_qa_distractor = _align_qa
process_sample_hotpot_qa_fullwiki = _align_qa
process_sample_newsqa = lambda x: x  # samples are processed when loaded
process_sample_quail = _align_qa
process_sample_quoref = process_sample_squad_v2
process_sample_ropes = _align_qa
process_sample_sciq = _align_qa


def process_sample_strategy_qa(sample):
    return {
        "text_a": " ".join(sample["facts"]),
        "text_b": sample["question"],
        "label": int(sample["answer"]),
    }


process_sample_msmarco = _align_paraphrase
process_sample_wikihow = _align_summarization
process_sample_gap = _align_qa
