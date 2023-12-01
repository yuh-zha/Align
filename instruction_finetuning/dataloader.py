import json
import logging
import random
from typing import Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from torch.utils.data import Dataset
from flan_prompts import PROMPT_OPTIONS, FLAN_PROMPTS
from align.dataloader import DSTDataLoader


class FLANDataSet(Dataset):
    def __init__(self, dataset, model_name='bert-base-uncased', tokenizer_max_length=512) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_max_length = tokenizer_max_length
        self.config = AutoConfig.from_pretrained(model_name)
        self.dataset_type_dict = dict()

        self.dataset = dataset

        self.dataset_type_dict_init()
    
    def dataset_type_dict_init(self):
        for i, example in enumerate(self.dataset):
            try:
                self.dataset_type_dict[example['task']].append(i)
            except:
                self.dataset_type_dict[example['task']] = [i]

    def process_mnli(self, index):
        premise = self.dataset[index]['text_a']
        hypothesis = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['mnli'])])
        answer = chr(65 + self.dataset[index]['orig_label'])

        template = random.choice(FLAN_PROMPTS['mnli'])

        input_ids = self.tokenizer(template[0].format(premise=premise, hypothesis=hypothesis, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        
        # print(template[0].format(premise=premise, hypothesis=hypothesis, options_=options_), template[1].format(answer=answer))
        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_snli(self, index):
        premise = self.dataset[index]['text_a']
        hypothesis = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['snli'])])
        answer = chr(65 + self.dataset[index]['orig_label'])

        template = random.choice(FLAN_PROMPTS['snli'])

        input_ids = self.tokenizer(template[0].format(premise=premise, hypothesis=hypothesis, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_rte(self, index):
        premise = self.dataset[index]['text_a']
        hypothesis = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['rte'])])
        answer = chr(66 - self.dataset[index]['orig_label'])# 65 + 1 - xx # generated dataset reversed order 0: not entail, 1: entail

        template = random.choice(FLAN_PROMPTS['rte'])

        input_ids = self.tokenizer(template[0].format(premise=premise, hypothesis=hypothesis, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_anli(self, index):
        context = self.dataset[index]['text_a']
        hypothesis = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['anli'])])
        answer = chr(65 + self.dataset[index]['orig_label'])

        template = random.choice(FLAN_PROMPTS['anli'])

        input_ids = self.tokenizer(template[0].format(context=context, hypothesis=hypothesis, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_paws_wiki(self, index):
        sentence1 = self.dataset[index]['text_a']
        sentence2 = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['paws_wiki'])])
        answer = chr(65 + self.dataset[index]['orig_label'])

        template = random.choice(FLAN_PROMPTS['paws_wiki'])

        input_ids = self.tokenizer(template[0].format(sentence1=sentence1, sentence2=sentence2, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_glue_qqp(self, index):
        question1 = self.dataset[index]['text_a']
        question2 = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['glue_qqp'])])
        answer = chr(65 + self.dataset[index]['orig_label'])

        template = random.choice(FLAN_PROMPTS['glue_qqp'])

        input_ids = self.tokenizer(template[0].format(question1=question1, question2=question2, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_squad_v1(self, index):
        context = self.dataset[index]['text_a']
        question = self.dataset[index]['text_b'][0]
        answer = self.dataset[index]['text_c'][0]

        template = random.choice(FLAN_PROMPTS['squad_v1'])

        input_ids = self.tokenizer(template[0].format(context=context, question=question),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )
    
    def process_squad_v2(self, index):
        context = self.dataset[index]['text_a']
        question = self.dataset[index]['text_b'][0]
        answer = self.dataset[index]['text_c'][0]

        template = random.choice(FLAN_PROMPTS['squad_v2'])

        input_ids = self.tokenizer(template[0].format(context=context, question=question),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )
    
    def process_openbookqa(self, index):
        fact = self.dataset[index]['text_a']
        question = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(self.dataset[index]['text_c'])])
        try:
            answer = chr(int(self.dataset[index]['orig_label']) + 65)
        except:
            answer = self.dataset[index]['orig_label']

        template = random.choice(FLAN_PROMPTS['openbookqa'])

        input_ids = self.tokenizer(template[0].format(fact=fact, question=question, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_bool_q(self, index):
        text = self.dataset[index]['text_a']
        question = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['bool_q'])])
        answer = chr(int(self.dataset[index]['orig_label']) + 65)

        template = random.choice(FLAN_PROMPTS['bool_q'])

        input_ids = self.tokenizer(template[0].format(text=text, question=question, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_gap(self, index):
        context = self.dataset[index]['text_a']
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in self.dataset[index]['text_b']])
        answer = self.dataset[index]['orig_label']

        template = random.choice(FLAN_PROMPTS['gap'])

        input_ids = self.tokenizer(template[0].format(context=context, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )
    
    def process_xsum(self, index):
        text = self.dataset[index]['text_a']
        summary = self.dataset[index]['text_b'][0]

        template = random.choice(FLAN_PROMPTS['xsum'])

        input_ids = self.tokenizer(template[0].format(text=text),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(summary=summary), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_msmarco(self, index):
        context = self.dataset[index]['text_a']
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['msmarco'])])
        answer = chr(self.dataset[index]['orig_label'] + 65)

        template = random.choice(FLAN_PROMPTS['msmarco'])

        input_ids = self.tokenizer(template[0].format(context=context, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )
    
    def process_stsb(self, index):
        sentence1 = self.dataset[index]['text_a']
        sentence2 = self.dataset[index]['text_b'][0]
        options_ = '\n'.join([f"({chr(i+65)}) "+each for i, each in enumerate(PROMPT_OPTIONS['stsb'])])

        score  = self.dataset[index]['orig_label'] 
        if self.dataset[index]['orig_label'] > 1.0:
            score = 1.0
        elif self.dataset[index]['orig_label'] < 0.0:
            score = 0.0
        answer = chr(int(score * 5) + 65)

        template = random.choice(FLAN_PROMPTS['stsb'])

        input_ids = self.tokenizer(template[0].format(sentence1=sentence1, sentence2=sentence2, options_=options_),
                                            padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        label = self.tokenizer(template[1].format(answer=answer), 
                                        padding='max_length', max_length=self.tokenizer_max_length, truncation=True).input_ids
        

        return (
            torch.tensor(input_ids), 
            torch.tensor(label)
        )

    def process_ctc(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        reg_label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(-100), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(reg_label) # reg label, float
        )

    def __getitem__(self, index):
        if self.dataset[index]['flan'] == 'mnli':
            input_ids, label = self.process_mnli(index)
        elif self.dataset[index]['flan'] == 'snli':
            input_ids, label = self.process_snli(index)
        elif self.dataset[index]['flan'] == 'rte':
            input_ids, label = self.process_rte(index)
        elif self.dataset[index]['flan'] == 'anli':
            input_ids, label = self.process_anli(index)
        elif self.dataset[index]['flan'] == 'paws_wiki':
            input_ids, label = self.process_paws_wiki(index)
        elif self.dataset[index]['flan'] == 'glue_qqp':
            input_ids, label = self.process_glue_qqp(index)
        elif self.dataset[index]['flan'] == 'squad_v1':
            input_ids, label = self.process_squad_v1(index)
        elif self.dataset[index]['flan'] == 'squad_v2':
            input_ids, label = self.process_squad_v2(index)
        elif self.dataset[index]['flan'] == 'openbookqa':
            input_ids, label = self.process_openbookqa(index)
        elif self.dataset[index]['flan'] == 'bool_q':
            input_ids, label = self.process_bool_q(index)
        elif self.dataset[index]['flan'] == 'gap':
            input_ids, label = self.process_gap(index)
        elif self.dataset[index]['flan'] == 'xsum':
            input_ids, label = self.process_xsum(index)
        elif self.dataset[index]['flan'] == 'msmarco':
            input_ids, label = self.process_msmarco(index)
        elif self.dataset[index]['flan'] == 'stsb':
            input_ids, label = self.process_stsb(index)
        else:
            print(f"unsupported: {self.dataset[index]['flan']}")
            exit()


        return {
            'input_ids': input_ids,
            'labels': label
            }
        

    def __len__(self):
        return len(self.dataset)
    

class FLANDataLoader(DSTDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            print("Already Initilized LightningDataModule!")
            return
        
        self.init_training_set()

        self.dataset = dict()
        if not self.is_finetune:
            self.dataset['train'] = FLANDataSet(dataset=self.raw_dataset[:int(self.train_eval_split*len(self.raw_dataset))], model_name=self.model_name)
            self.dataset['test'] = FLANDataSet(dataset=self.raw_dataset[int(self.train_eval_split*len(self.raw_dataset)):], model_name=self.model_name)
        else:
            self.dataset['train'] = FLANDataSet(dataset=self.raw_dataset[:], model_name=self.model_name)
            self.dataset['test'] = FLANDataSet(dataset=self.val_raw_dataset[:], model_name=self.model_name)
            
    
    def init_training_set(self):
        self.raw_dataset = []
        if self.sample_mode == 'seq':
            for each_dataset in self.dataset_config:
                dataset_length = sum([1 for line in open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8')])
                dataset_length_limit = self.dataset_config[each_dataset]['size'] if isinstance(self.dataset_config[each_dataset]['size'], int) else int(self.dataset_config[each_dataset]['size'] * dataset_length)
                with open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    try:
                        for i, example in enumerate(f):
                            if i >= dataset_length_limit:
                                break
                            json_obj = json.loads(example)
                            json_obj['flan'] = self.dataset_config[each_dataset]['flan']
                            self.raw_dataset.append(json_obj) ## + dataset_name
                    except:
                        print(f"failed to load data from {each_dataset}.json, exiting...")
                        exit()
            
            random.shuffle(self.raw_dataset)
        
        elif self.sample_mode == 'proportion':
            for each_dataset in tqdm(self.dataset_config, desc="Loading data from disk..."):
                with open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    try:
                        for i, example in enumerate(f):
                            jsonobj = json.loads(example)
                            jsonobj['dataset_name'] = each_dataset
                            self.raw_dataset.append(jsonobj) ## + dataset_name
                    except:
                        print(f"failed to load data from {each_dataset}.json, exiting...")
                        exit()
            
            random.shuffle(self.raw_dataset)
        
        if self.is_finetune:
            self.val_raw_dataset = []
            for each_dataset in self.val_dataset_config:
                dataset_length = sum([1 for line in open(self.val_dataset_config[each_dataset]['data_path'], 'r', encoding='utf8')])
                dataset_length_limit = self.val_dataset_config[each_dataset]['size'] if isinstance(self.val_dataset_config[each_dataset]['size'], int) else int(self.val_dataset_config[each_dataset]['size'] * dataset_length)
                with open(self.val_dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    for i, example in enumerate(f):
                        if i >= dataset_length_limit:
                            break
                        self.val_raw_dataset.append(json.loads(example))
            
            random.shuffle(self.val_raw_dataset)