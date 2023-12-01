import json
from tqdm import tqdm
import json
import itertools


def combine_subsets(subsets, output_path):
    datasets = {}
    for name, path in subsets.items():
        with open(path, 'r', encoding='utf8') as file:
            datasets[name] = json.load(file)
    for name, dataset in datasets.items():
        for sample in dataset['data']:
            for paragraph in sample['paragraphs']:
                for qa in paragraph['qas']:
                    qa['id'] = f'{name}/{qa["id"]}'
    joined = {
        'version': next(iter(datasets.values()))['version'],
        'data': list(itertools.chain.from_iterable([subset['data'] for subset in datasets.values()]))
    }
    with open(output_path, 'w', encoding='utf8') as output:
        json.dump(joined, output)


def combine_predictions(prediction_files, output_path):
    combined_predictions = {}
    for name, path in prediction_files.items():
        with open(path, 'r', encoding='utf8') as file:
            predictions = json.load(file)
            combined_predictions.update({
                f'{name}/{key}': val
                for key, val in predictions.items()
            })
    with open(output_path, 'w', encoding='utf8') as output:
        json.dump(combined_predictions, output)



def load_json_dataset(path):
    data = []
    with open(path, encoding="utf-8") as f:
        squad = json.load(f)
        for example in squad["data"]:
            title = example.get("title", "")
            for paragraph in example["paragraphs"]:
                # do not strip leading blank spaces GH-2585
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"]
                                     for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]

                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    data.append({
                        "title": title,
                        "context": context,
                        "question": question,
                        "id": id_,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                    })
    return data


class AlignReranker:
    def __init__(self, align_ckpt, align_model, device) -> None:
        from data_utils.generate_training_data import QA2D
        from inference import Inferencer

        self.align = Inferencer(
            ckpt_path=align_ckpt, model=align_model, device=device[0], verbose=False)
        self.align.nlg_eval_mode = 'nli_sp'
        self.qa2d = QA2D(device=device[1], verbose=False)

    def get_align_score(self, context, question, candidate):
        hypo = self.qa2d.generate([question], [candidate])
        return self.align.nlg_eval([context], hypo)[1][0].item()

    def get_align_score_concat(self, context, question, candidate):
        hypo = question + ' ' + candidate
        return self.align.inference([context], [hypo])[1][0].item()


def get_align_na(dataset, prediction_path, na_save_path, align_fn):
    with open(prediction_path, 'r', encoding='utf8') as prediction_file:
        predictions = json.load(prediction_file)
    na_prob = {}
    for sample in tqdm(dataset):
        prediction = predictions[sample['id']]
        if not prediction:
            na_prob[sample['id']] = 1.
        else:
            na_prob[sample['id']] = 1. - \
                align_fn(sample['context'], sample['question'], prediction)
    with open(na_save_path, 'w', encoding='utf8') as output:
        json.dump(na_prob, output)

def apply_na_threshold(prediction_path, na_path, output_path, threshold):
    with open(prediction_path, 'r', encoding='utf8') as prediction_file:
        predictions = json.load(prediction_file)
    with open(na_path, 'r', encoding='utf8') as na_prob_file:
        na_prob = json.load(na_prob_file)
    for sample_id, p in na_prob.items():
        if p > threshold:
            predictions[sample_id] = ''
    with open(output_path, 'w', encoding='utf8') as output:
        json.dump(predictions, output)
