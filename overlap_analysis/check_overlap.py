from nltk.tokenize import word_tokenize
import jsonlines
from tqdm import tqdm
from itertools import islice
from datasets import load_dataset
from overlapy import OverlapyTestSet, Overlapy
import pickle
import pandas as pd
import random
from data.eval.summac.benchmark import SummaCBenchmark
import json
import os

random.seed(42)

ALL_TRAINING_DATASETS = {
    ### NLI
    'mnli': {'task_type': 'nli', 'data_path': 'data/training/mnli.json'},     
    'doc_nli': {'task_type': 'bin_nli', 'data_path': 'data/training/doc_nli.json'},
    'snli': {'task_type': 'nli', 'data_path': 'data/training/snli.json'},
    'anli_r1': {'task_type': 'nli', 'data_path': 'data/training/anli_r1.json'},
    'anli_r2': {'task_type': 'nli', 'data_path': 'data/training/anli_r2.json'},
    'anli_r3': {'task_type': 'nli', 'data_path': 'data/training/anli_r3.json'},

    ### fact checking
    'nli_fever': {'task_type': 'fact_checking', 'data_path': 'data/training/nli_fever.json'},
    'vitaminc': {'task_type': 'fact_checking', 'data_path': 'data/training/vitaminc.json'},

    ### paraphrase
    'paws': {'task_type': 'paraphrase', 'data_path': 'data/training/paws.json'},
    'paws_qqp': {'task_type': 'paraphrase', 'data_path': 'data/training/paws_qqp.json'},
    'paws_unlabeled': {'task_type': 'paraphrase', 'data_path': 'data/training/paws_unlabeled.json'},
    'qqp': {'task_type': 'paraphrase', 'data_path': 'data/training/qqp.json'},
    'wiki103': {'task_type': 'paraphrase', 'data_path': 'data/training/wiki103.json'},

    ### QA
    'squad_v2': {'task_type': 'qa', 'data_path': 'data/training/squad_v2_new.json'},
    'race': {'task_type': 'qa', 'data_path': 'data/training/race.json'},
    'adversarial_qa': {'task_type': 'qa', 'data_path': 'data/training/adversarial_qa.json'},
    'drop': {'task_type': 'qa', 'data_path': 'data/training/drop.json'},
    'hotpot_qa_distractor': {'task_type': 'qa', 'data_path': 'data/training/hotpot_qa_distractor.json'},
    'hotpot_qa_fullwiki': {'task_type': 'qa', 'data_path': 'data/training/hotpot_qa_fullwiki.json'},
    'newsqa': {'task_type': 'qa', 'data_path': 'data/training/newsqa.json'},
    'quoref': {'task_type': 'qa', 'data_path': 'data/training/quoref.json'},
    'ropes': {'task_type': 'qa', 'data_path': 'data/training/ropes.json'},
    'boolq': {'task_type': 'qa', 'data_path': 'data/training/boolq.json'},
    'eraser_multi_rc': {'task_type': 'qa', 'data_path': 'data/training/eraser_multi_rc.json'},
    'quail': {'task_type': 'qa', 'data_path': 'data/training/quail.json'},
    'sciq': {'task_type': 'qa', 'data_path': 'data/training/sciq.json'},
    'strategy_qa': {'task_type': 'qa', 'data_path': 'data/training/strategy_qa.json'},

    ### Coreference
    'gap': {'task_type': 'coreference', 'data_path': 'data/training/gap.json'},

    ### Summarization
    'wikihow': {'task_type': 'summarization', 'data_path': 'data/training/wikihow.json'},

    ### Information Retrieval
    'msmarco': {'task_type': 'ir', 'data_path': 'data/training/msmarco.json'},

    ### STS
    'stsb': {'task_type': 'sts', 'data_path': 'data/training/stsb.json'},
    'sick': {'task_type': 'sts', 'data_path': 'data/training/sick.json'},
}

def load_huggingface_dataset(*args):
    return load_dataset(*args[:-1])[args[-1]]

def load_paws_qqp(path):
    df = pd.read_csv(path, sep='\t')
    return [
        {"sentence1": row['sentence1'][2:-1], 'sentence2': row['sentence2'][2:-1], 'label': row['label']}
        for _, row in df.iterrows()
    ]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_sample_text(sample, fields):
    items = []
    for field in fields:
        if isinstance(field, tuple):
            items.append(get_sample_text(sample[field[0]], field[1:]))
        elif isinstance(sample[field], (list, tuple)):
            items.extend(sample[field])
        else:
            items.append(sample[field])
    return ' '.join(items)

TRAIN_DATA_SIZE = 500000

TEST_DATASETS = {
    'mnli_matched': (load_huggingface_dataset('multi_nli', 'validation_matched'), ('premise', 'hypothesis')),
    'mnli_mismatched': (load_huggingface_dataset('multi_nli', 'validation_mismatched'), ('premise', 'hypothesis')),
    'anli_1': (load_huggingface_dataset('anli', 'test_r1'), ('premise', 'hypothesis')),
    'anli_2': (load_huggingface_dataset('anli', 'test_r2'), ('premise', 'hypothesis')),
    'anli_3': (load_huggingface_dataset('anli', 'test_r3'), ('premise', 'hypothesis')),
    'snli': (load_huggingface_dataset('snli', 'test'), ('premise', 'hypothesis')),
    'nli_fever': (load_huggingface_dataset('pietrolesci/nli_fever', 'dev'), ('premise', 'hypothesis')),
    'vitaminc': (load_huggingface_dataset('tals/vitaminc', 'test'), ('evidence', 'claim')),
    'sick': (load_huggingface_dataset('sick', 'default', 'test'), ('sentence_A', 'sentence_B')),
    'stsb': (load_huggingface_dataset('glue', 'stsb', 'validation'), ('sentence1', 'sentence2')),
    'paws': (load_huggingface_dataset('paws', 'labeled_final', 'test'), ('sentence1', 'sentence2')),
    'paws_qqp': (load_paws_qqp('/data/ruichen/General_Alignment_Model/paws_qqp/output/dev_and_test.tsv'), ('sentence1', 'sentence2')),
    'qqp': (load_huggingface_dataset('glue', 'qqp', 'validation'), ('question1', 'question2')),
    'race_m': (load_huggingface_dataset('race', 'middle', 'test'), ('article', 'question', 'options')),
    'race_h': (load_huggingface_dataset('race', 'high', 'test'), ('article', 'question', 'options')),
    'eraser_multi_rc': (load_huggingface_dataset('eraser_multi_rc', 'test'), ('passage', 'query_and_answer')),
    'boolq': (load_huggingface_dataset('boolq', 'validation'), ('passage', 'question')),
    'quail': (load_huggingface_dataset('quail', 'challenge'), ('context', 'question', 'answers')),
    'sciq': (load_huggingface_dataset('sciq', 'test'), ('support', 'question', 'correct_answer', 'distractor1', 'distractor2', 'distractor3')),
    'gap': (load_huggingface_dataset('gap', 'test'), ('Text', 'A', 'B')),

    'axb': (load_huggingface_dataset('super_glue', 'axb', 'test'), ('sentence1', 'sentence2')),
    'axg': (load_huggingface_dataset('super_glue', 'axg', 'test'), ('premise', 'hypothesis')),
    'cb': (load_huggingface_dataset('super_glue', 'cb', 'validation'), ('premise', 'hypothesis')),
    'rte': (load_huggingface_dataset('super_glue', 'rte', 'validation'), ('premise', 'hypothesis')),
    'wnli': (load_huggingface_dataset('SetFit/wnli', 'validation'), ('text1', 'text2')),
    'sem_eval': (load_huggingface_dataset('sem_eval_2014_task_1', 'test'), ('premise', 'hypothesis')),
    'mrpc': (load_huggingface_dataset('glue', 'mrpc', 'test'), ('sentence1', 'sentence2')),
    'dream': (load_huggingface_dataset('dream', 'test'), ('dialogue', 'question', 'choice')),
    'quartz': (load_huggingface_dataset('quartz', 'test'), ('para', 'question', ('choices', 'text'))),

    'xsumfaith': (load_json('data/eval/xsumfaith.json'), ('document', 'claim')),
    'summeval': (load_json('data/eval/summeval.json'), ('document', 'summary')),
    'qags_xsum': (load_json('data/eval/qags_xsum.json'), ('document', 'summary')),
    'qags_cnndm': (load_json('data/eval/qags_cnndm.json'), ('document', 'summary')),
    'frank': (load_json('data/eval/frank.json'), ('document', 'summary')),
    'samsum': (load_json('data/eval/samsum.json'), ('article', 'summary')),
}

summac_benchmark = SummaCBenchmark(benchmark_folder="./data/eval/summac/benchmark", cut='test')
for dataset in summac_benchmark.datasets:
    TEST_DATASETS[f"summac_{dataset['name']}"] = (dataset['dataset'], ('document', 'claim'))

for file_name in os.listdir('data/eval/true'):
    if 'qags' in file_name:
        name = 'true_' + '_'.join(file_name.split('_')[:2])
    else:
        name = 'true_' + '_'.join(file_name.split('_')[:1])
    df = pd.read_csv(os.path.join('data/eval/true', file_name))
    TEST_DATASETS[name] = (df.to_dict(orient="records"), ('grounding', 'generated_text'))

N_SAMPLES = 1000

if __name__ == '__main__':
    train_samples = []
    for name, setting in tqdm(ALL_TRAINING_DATASETS.items(), desc='tokenizing training set'):
        with jsonlines.open(setting['data_path'], 'r') as dataset:
            for sample in tqdm(islice(dataset, TRAIN_DATA_SIZE), total=TRAIN_DATA_SIZE, desc=name):
                sample_text = ' '.join((sample['text_a'], *sample['text_b'], *sample['text_c']))
                train_samples.append(word_tokenize(sample_text))

    for name, (dataset, fields) in tqdm(TEST_DATASETS.items(), desc='processing test set'):
        index_map, subset = list(zip(*random.sample(list(enumerate(dataset)), min(len(dataset), N_SAMPLES))))
        samples = [
            word_tokenize(get_sample_text(sample, fields))
            for sample in tqdm(subset, desc=f'tokenizing {name}')
        ]

        with open(f'n_gram_overlap/test_set_matches/{name}.index_map.pickle', 'wb') as file:
            pickle.dump(index_map, file)

        test_dataset = OverlapyTestSet(name, examples=samples)
        overlapy = Overlapy(testsets=[test_dataset], dataset=train_samples, n_workers=32)

        matches = overlapy.run()
        with open(f'n_gram_overlap/test_set_matches/{test_dataset.name}.matches.pickle', 'wb') as file:
            pickle.dump(matches, file)

        test_set_matches = list(test_dataset.get_matches(matches))
        with open(f'n_gram_overlap/test_set_matches/{test_dataset.name}.samples.pickle', 'wb') as file:
            pickle.dump(test_set_matches, file)

