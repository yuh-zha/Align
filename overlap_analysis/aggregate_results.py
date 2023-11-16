import pickle
import json
import pandas as pd
import numpy as np

DATASETS = [
    'mnli_mismatched', 'mnli_matched', 'anli_1', 'anli_2', 'anli_3', 'snli','nli_fever', 'vitaminc', 'sick', 'stsb', 'paws', 'paws_qqp', 'qqp',
    'race_m', 'race_h', 'eraser_multi_rc', 'boolq', 'quail', 'sciq', 'gap',
    'axb', 'axg', 'cb', 'rte', 'wnli', 'sem_eval', 'mrpc', 'dream', 'quartz',
    'xsumfaith', 'summeval', 'qags_xsum', 'qags_cnndm', 'frank-xsum', 'frank-cnndm', 'samsum', 
    'summac_cogensumm', 'summac_xsumfaith', 'summac_polytope', 
    'summac_factcc', 'summac_summeval', 'summac_frank',
    'true_frank', 'true_summeval', 'true_mnbm', 'true_qags_cnndm', 'true_qags_xsum',
    'true_begin', 'true_q2', 'true_dialfact', 'true_paws', 'true_fever', 'true_vitc',
]

SPLITS = ['nlg_eval', 'nlu']
SUBSETS = ['all', 'clean', 'dirty']

NAME_MAP = {
    'anli-1': 'anli_1',
    'anli-2': 'anli_2',
    'anli-3': 'anli_3',
    'sick-r': 'sick',
    'race-m': 'race_m',
    'race-h': 'race_h',
}

def get_metric(dataset: str):
    if dataset.startswith('true_') or dataset.startswith('summac_'):
        return 'AUC'
    elif dataset in (
        'sick', 'stsb',
        'xsumfaith', 'summeval', 'qags_xsum', 'qags_cnndm', 'frank-xsum', 'frank-cnndm', 'samsum'
    ):
        return 'Pearson'
    else:
        return 'Accuracy'

statistics = []
for split in SPLITS:
    with open(f'n_gram_overlap/test_set_matches/{split}_statistics.jsonl', 'r') as file:
        statistics.extend(json.load(file))

# dirty fix
frank_result = next(entry for entry in statistics if entry['dataset'] == 'frank')
statistics_fix = [entry for entry in statistics if entry['dataset'] != 'frank'] + [
    {**frank_result, 'dataset': 'frank-xsum'},
    {**frank_result, 'dataset': 'frank-cnndm'}, 
]

statistics_df = pd.DataFrame(statistics_fix)

all_results = []
for split in SPLITS:
    result_df = None
    for subset in SUBSETS:
        with open(f'exp_results/overlap/roberta-large_{subset}_{split}.pkl', 'rb') as file:
            results = pickle.load(file)
            names = []
            metrics = []
            for entry in results:
                name = NAME_MAP.get(entry['Dataset_name'], entry['Dataset_name'])
                if name not in DATASETS:
                    print("name not in DATASETS", name)
                    continue
                names.append(name)
                metrics.append(entry[get_metric(name)][0])
            df = pd.DataFrame({'dataset': names, subset: metrics})
            result_df = df if result_df is None else result_df.merge(df, how='outer', on='dataset')
    all_results.append(result_df)
result_df = pd.concat(all_results)

complete_data = statistics_df.merge(result_df, how='outer', on='dataset')
complete_data.dataset = pd.Categorical(complete_data.dataset, categories=DATASETS)
complete_data.sort_values('dataset', inplace=True)

# dirty fix
complete_data.loc[complete_data['dataset'] == 'cb', 'dirty'] = np.nan

print(complete_data)
assert (complete_data['dirty'].isna() == ((complete_data['dirty_count'] < 100) & (complete_data['dirty_percent'] <= 0.5))).all()
assert (complete_data['clean'].isna() == ((complete_data['clean_count'] < 100) & (complete_data['dirty_percent'] >= 0.5))).all()
