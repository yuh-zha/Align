from evaluate import Evaluator, FLANEvaluator
from baselines import *
from align.inference import Inferencer
import time
import json
import os
from argparse import ArgumentParser

SAVE_ALL_TABLES = True
SAVE_AND_PRINT_TIMER = False

class Timer():
    def __init__(self) -> None:
        self.t0 = time.time()
        self.save_path = 'exp_results/time.json'
    
    def finish(self, display_name):
        t1 = time.time()
        time_pass = t1 - self.t0
        if SAVE_AND_PRINT_TIMER:
            print(f"Evalautor {display_name} finished in {time_pass} secs.")
            with open(self.save_path, 'a', encoding='utf8') as f:
                json.dump({display_name: time_pass}, f)
                f.write('\n')


def eval_ctc(model_type, tasks):
    ctc_scorer = CTCScorer(model_type)
    evaluator = Evaluator(eval_tasks=tasks, align_func=ctc_scorer.score, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/CTC-{model_type}"

    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"CTC-{model_type}")

def eval_simcse(model_type, device, tasks):
    simcse_scorer = SimCSEScorer(model_type, device)
    evaluator = Evaluator(eval_tasks=tasks, align_func=simcse_scorer.score, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/{model_type.split('/')[-1]}_f"

    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"{model_type.split('/')[-1]}_f")

def eval_bleurt(checkpoint, tasks):
    bleurt_scorer = BleurtScorer(checkpoint)
    evaluator = Evaluator(eval_tasks=tasks, align_func=bleurt_scorer.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/BLEURT"

    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"BLEURT")

def eval_bertscore(model_type, device, batch_size, tasks):
    bertscore_scorer = BertScoreScorer(model_type=model_type, metric='f1', device=device, batch_size=batch_size)
    evaluator = Evaluator(eval_tasks=tasks, align_func=bertscore_scorer.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/bertscore_{model_type.replace('/', '-')}_f"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"bertscore_{model_type.replace('/', '-')}_f")

def eval_bartscore(checkpoint, device, tasks):
    bartscore_scorer = BartScoreScorer(checkpoint, device)
    evaluator = Evaluator(eval_tasks=tasks, align_func=bartscore_scorer.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/bartscore-{checkpoint.replace('/','-')}"

    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"bartscore-{checkpoint.replace('/','-')}")

### Below are Baselines for SummaC
def eval_mnli(tasks, model="roberta-large-mnli", device='cuda:0'):
    mnli_scorer = MNLIScorer(model=model, device=device)
    evaluator = Evaluator(eval_tasks=tasks, align_func=mnli_scorer.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/mnli-{model}"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"mnli-{model}")

def eval_ner(tasks):
    ner_scorer = NERScorer()
    evaluator = Evaluator(eval_tasks=tasks, align_func=ner_scorer.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/NER"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"NER")

def eval_unieval(tasks, device='cuda:0'):
    unieval = UniEvalScorer(task='fact', device=device)
    evaluator = Evaluator(eval_tasks=tasks, align_func=unieval.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/UniEval"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"UniEval")

def eval_feqa(tasks):
    feqa = FEQAScorer()
    evaluator = Evaluator(eval_tasks=tasks, align_func=feqa.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/FEQA"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"FEQA")

def eval_questeval(tasks):
    questeval = QuestEvalScorer()
    evaluator = Evaluator(eval_tasks=tasks, align_func=questeval.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/QuestEval"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"QuestEval")

def eval_qafacteval(tasks, device='cuda:0'):
    import os, sys
    warning("using conda env qaeval!!!")
    qafacteval = QAFactEvalScorer(device=device, model_folder=os.path.abspath('../BaselineForNLGEval/QAFactEval/models'))
    evaluator = Evaluator(eval_tasks=tasks, align_func=qafacteval.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/QAFactEval"
    evaluator.evaluate()

def eval_dae(tasks, model_dir=None, device=0):
    dae = DAEScorer(model_dir=model_dir, device=device)
    evaluator = Evaluator(eval_tasks=tasks, align_func=dae.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/DAE"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"DAE")

def eval_bleu(tasks, n_grams=1):
    bleu = BLEUScorer(n_grams=n_grams)
    evaluator = Evaluator(eval_tasks=tasks, align_func=bleu.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/BLEU-{n_grams}"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"BLEU-{n_grams}")

def eval_rouge(tasks, rouge_type='1'):
    rouge = ROUGEScorer(rouge_type=rouge_type)
    evaluator = Evaluator(eval_tasks=tasks, align_func=rouge.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/ROUGE-{rouge_type}"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"ROUGE-{rouge_type}")

def eval_factcc(script_path, test_data_path,result_path, tasks):
    factcc = FactCCScorer(script_path=script_path, test_data_path=test_data_path, result_path=result_path)
    evaluator = Evaluator(eval_tasks=tasks, align_func=factcc.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/FactCC"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"FactCC")

def eval_blanc(tasks, device='cuda:0', batch_size=64):
    blanc = BLANCScorer(device=device, batch_size=batch_size)
    evaluator = Evaluator(eval_tasks=tasks, align_func=blanc.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/BLANC"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"BLANC")

def eval_summac(tasks, summac_type='conv', device='cuda:0'):
    summac = SummaCScorer(summac_type=summac_type, device=device)
    evaluator = Evaluator(eval_tasks=tasks, align_func=summac.scorer, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"baselines/SummaC-{summac_type}"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"SummaC-{summac_type}")

def eval_align_nlg(ckpt_path, tasks, comment='', base_model='roberta-large', batch_size=32, device='cuda:0', nlg_eval_mode='nli_sp'):
    align = Inferencer(ckpt_path=ckpt_path, model=base_model, batch_size=batch_size, device=device)
    align.nlg_eval_mode = nlg_eval_mode

    evaluator = Evaluator(eval_tasks=tasks, align_func=align.nlg_eval, save_all_tables=SAVE_ALL_TABLES)
    name = f'Align-{nlg_eval_mode}-{base_model}'
    if comment:
        name += '_' + comment
    evaluator.result_save_name = f"align_facutal_consistency/{name}"

    timer = Timer()
    evaluator.evaluate()
    timer.finish(name)

def eval_align(tasks, ckpt_path, save_comment="", base_model='roberta-large', batch_size=32, device='cuda:0'):
    align = Inferencer(ckpt_path=ckpt_path, model=base_model, batch_size=batch_size, device=device)

    evaluator = Evaluator(eval_tasks=tasks, align_func=align.inference, save_all_tables=SAVE_ALL_TABLES)
    evaluator.result_save_name = f"align_nlu/{save_comment}{ckpt_path.split('/')[-1].split('.ckpt')[0]}" 
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(f"ALIGN-NLU-{base_model}")

def eval_flan(tasks, device='cuda:0', model_name='google/flan-t5-base', batch_size=32, custom_ckpt=None):
    flan_t5 = FLANScorer(device=device, model_name=model_name, batch_size=batch_size, custom_ckpt=custom_ckpt)
    evaluator = FLANEvaluator(eval_tasks=tasks, align_func=flan_t5.scorer, save_all_tables=SAVE_ALL_TABLES)
    name = f"FLAN-{model_name.replace('/', '-')}"
    if custom_ckpt is not None:
        name += custom_ckpt.split('/')[-1].split('.')[0]
    evaluator.result_save_name = f"baselines/{name}"
    
    timer = Timer()
    evaluator.evaluate()
    timer.finish(name)

def run_benchmarks(args, argugment_error):
    os.makedirs('exp_results/baselines', exist_ok=True)
    os.makedirs('exp_results/align_nlu', exist_ok=True)
    os.makedirs('exp_results/align_facutal_consistency', exist_ok=True)

    if args.align_nlu:
        if not all((args.align_model, args.align_ckpt)):
            argugment_error('--align-model, and --align-ckpt must be specified to run Align NLU benchmark')
        eval_align(
            ckpt_path=args.align_ckpt,
            save_comment=args.align_nlu_comment,
            base_model=args.align_model,
            device=args.device,
            tasks=args.nlu_tasks
        )

    if args.align_factual_consistency:
        if not all((args.align_model, args.align_ckpt)):
            argugment_error('--align-model, and --align-ckpt must be specified to run Align factual consistency benchmark')
        eval_align_nlg(
            nlg_eval_mode=args.align_fact_eval_mode, 
            ckpt_path=args.align_ckpt, 
            base_model=args.align_model, 
            device=args.device,
            tasks=args.factual_consistency_tasks,
            comment=args.align_fact_comment
        )

    if args.flan_t5:
        if not args.flan_t5_model:
            argugment_error('--flan-t5-model must be specified to run FLAN T5 NLU benchmark')
        eval_flan(
            tasks=args.nlu_tasks,
            device=args.device,
            model_name=args.flan_t5_model,
            custom_ckpt=args.flan_t5_custom_ckpt
        )

    if args.ctc:
        if not args.ctc_type:
            argugment_error('--ctc-type must be specified to run CTC baseline')
        for type in args.ctc_type:
            eval_ctc(type, tasks=args.factual_consistency_tasks)

    if args.simcse:
        if not args.simcse_ckpt:
            argugment_error('--simcse-ckpt must be specified to run SimCSE baseline')
        for ckpt in args.simcse_ckpt:
            eval_simcse(ckpt, device=args.device, tasks=args.factual_consistency_tasks)

    if args.bleurt:
        if not args.bleurt_ckpt:
            argugment_error('--bleurt-ckpt must be specified to run BLEURT baseline')
        eval_bleurt(args.bleurt_ckpt, tasks=args.factual_consistency_tasks)

    if args.bertscore:
        if not args.bertscore_ckpt or not args.bertscore_batch_size:
            argugment_error('--bertscore-ckpt and --bertscore-batch-size must be specified to run BERTScore baseline')
        for ckpt in args.bertscore_ckpt:
            eval_bertscore(ckpt, device=args.device, tasks=args.factual_consistency_tasks, batch_size=args.bertscore_batch_size)

    if args.bartscore:
        if not args.bartscore_ckpt:
            argugment_error('--bartscore-ckpt must be specified to run BARTScore baseline')
        for ckpt in args.bartscore_ckpt:
            eval_bartscore(ckpt, device=args.device, tasks=args.factual_consistency_tasks)

    if args.mnli:
        if not args.mnli_ckpt:
            argugment_error('--mnli-ckpt must be specified to run MNLI baseline')
        for ckpt in args.mnli_ckpt:
            eval_mnli(model=ckpt, device=args.device, tasks=args.factual_consistency_tasks)

    if args.ner:
        eval_ner(tasks=args.factual_consistency_tasks)

    if args.unieval:
        eval_unieval(tasks=args.factual_consistency_tasks, device=args.device)

    if args.feqa:
        eval_feqa(tasks=args.factual_consistency_tasks)

    if args.questeval:
        eval_questeval(tasks=args.factual_consistency_tasks)

    if args.qafacteval:
        eval_qafacteval(tasks=args.factual_consistency_tasks)

    if args.bleu:
        if not args.bleu_ngram:
            argugment_error('--bleu-ngram must be specified to run BLEU baseline')
        for n in args.bleu_ngram:
            eval_bleu(tasks=args.factual_consistency_tasks, n_grams=n)

    if args.rouge:
        if not args.rouge_type:
            argugment_error('--rouge-type must be specified to run ROUGE baseline')
        for type in args.rouge_type:
            eval_rouge(tasks=args.factual_consistency_tasks, rouge_type=type)

    if args.dae:
        if not args.dae_ckpt:
            argugment_error('--dae-ckpt must be specified to run DAE baseline')
        eval_dae(tasks=args.factual_consistency_tasks, model_dir=os.path.abspath(args.dae_ckpt))
    
    if args.factcc:
        if not all((args.factcc_script, args.factcc_test_data, args.factcc_result_path)):
            argugment_error('--factcc-script, --factcc-test-data, and --factcc-result-path must be specified to run FactCC baseline')
        eval_factcc(
            tasks=args.factual_consistency_tasks,
            script_path=os.path.abspath(args.factcc_script),
            test_data_path=os.path.abspath(args.factcc_test_data),
            result_path=os.path.abspath(args.factcc_result_path)
        )
    
    if args.blanc:
        if not args.blanc_batch_size:
            argugment_error('--blanc-batch-size must be specified to run BLANC baseline')
        eval_blanc(tasks=args.factual_consistency_tasks, device=args.device, batch_size=args.blanc_batch_size)

    if args.summac:
        if not args.summac_type:
            argugment_error('--summac-type must be specified to run SummaC baseline')
        for type in args.summac_type:
            eval_summac(tasks=args.factual_consistency_tasks, device=args.device, summac_type=type)


if __name__ == "__main__": 
    NLU_TASKS = [
        'mnli_mismatched', 'mnli_matched', 'nli_fever', 'axb', 'axg', 'cb', 'rte', 'wnli',  'sick', 'stsb',  'anli_1','anli_2', 'anli_3',
        'snli', 'vitaminc', 'mrpc', 'paws', 'sem_eval', 'paws_qqp', 'qqp', 'race_m', 'race_h', 'eraser_multi_rc', 'boolq', 'dream', 'quartz',
        'quail', 'sciq', 'gap'
    ]
    FACT_EVAL_TASKS = ['summac', 'true','xsumfaith', 'summeval', 'qags_xsum', 'qags_cnndm', 'newsroom', 'rank19', 'frank', 'samsum']

    parser = ArgumentParser()
    parser.add_argument('--nlu-tasks', nargs='+', type=str, default=NLU_TASKS, choices=NLU_TASKS)
    parser.add_argument('--factual-consistency-tasks', nargs='+', type=str, default=FACT_EVAL_TASKS, choices=FACT_EVAL_TASKS)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--timer', action='store_true', help='Time all metric runs')

    ### Align (ours)
    alignscore_parser = parser.add_argument_group('Align')
    alignscore_parser.add_argument('--align-model', type=str, choices=['roberta-base', 'roberta-large'])
    alignscore_parser.add_argument('--align-ckpt', type=str)
    # NLU
    alignscore_parser.add_argument('--align-nlu', action='store_true', help='Run Align NLU benchmark')
    alignscore_parser.add_argument('--align-nlu-comment', type=str, default='')
    # Factual consistency evaluation
    alignscore_parser.add_argument('--align-factual-consistency', action='store_true', help='Run Align factual consistency evaluation benchmark')
    alignscore_parser.add_argument(
        '--align-fact-eval-mode',
        type=str,
        choices=['bin', 'bin_sp', 'nli', 'nli_sp', 'reg', 'reg_sp', 'smart-n', 'smart-l'],
        default='nli_sp'
    )
    alignscore_parser.add_argument('--align-fact-comment', type=str, default='')

    # NLU baselines
    flan_t5_parser = parser.add_argument_group('NLU baseline - FLAN T5')
    flan_t5_parser.add_argument('--flan-t5', action='store_true', help='Run FLAN T5 baseline')
    flan_t5_parser.add_argument('--flan-t5-model', type=str)
    flan_t5_parser.add_argument('--flan-t5-custom-ckpt', type=str)

    ### Factual consistenct evaluation baselines
    ctc_parser = parser.add_argument_group('Factual consistency evaluation baseline - CTC')
    ctc_parser.add_argument('--ctc', action='store_true', help='Run CTC baseline')
    ctc_parser.add_argument(
        '--ctc-type',
        nargs='*',
        type=str,
        choices=['D-cnndm', 'E-roberta', 'R-cnndm'],
        default=['D-cnndm']
    )

    simcse_parser = parser.add_argument_group('Factual consistency evaluation baseline - SimCSE')
    simcse_models = [
        'princeton-nlp/unsup-simcse-bert-base-uncased',
        'princeton-nlp/unsup-simcse-bert-large-uncased',
        'princeton-nlp/unsup-simcse-roberta-base',
        'princeton-nlp/unsup-simcse-roberta-large',
        'princeton-nlp/sup-simcse-bert-base-uncased',
        'princeton-nlp/sup-simcse-bert-large-uncased',
        'princeton-nlp/sup-simcse-roberta-base',
        'princeton-nlp/sup-simcse-roberta-large'
    ]
    simcse_parser.add_argument('--simcse', action='store_true', help='Run SimCSE baseline')
    simcse_parser.add_argument(
        '--simcse-ckpt',
        nargs='*',
        type=str,
        choices=simcse_models,
        default=['princeton-nlp/sup-simcse-roberta-large']
    )

    bleurt_parser = parser.add_argument_group('Factual consistency evaluation baseline - BLEURT')
    bleurt_parser.add_argument('--bleurt', action='store_true', help='Run BLEURT baseline')
    bleurt_parser.add_argument('--bleurt-ckpt', type=str)

    bertscore_parser = parser.add_argument_group('Factual consistency evaluation baseline - BERTScore')
    bertscore_parser.add_argument('--bertscore', action='store_true', help='Run BERTScore baseline')
    bertscore_parser.add_argument(
        '--bertscore-ckpt',
        nargs='*',
        type=str,
            default=['microsoft/deberta-xlarge-mnli']
    )
    bertscore_parser.add_argument('--bertscore-batch-size', type=int, default=16)

    bartscore_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - BARTScore',
        description='Please clone https://github.com/neulab/BARTScore to baselines/BARTScore.'
    )
    bartscore_parser.add_argument('--bartscore', action='store_true', help='Run BARTScore baseline')
    bartscore_parser.add_argument(
        '--bartscore-ckpt',
        type=str,
        nargs='*',
        default=['facebook/bart-large-cnn']
    )

    mnli_parser = parser.add_argument_group('Factual consistency evaluation baseline - MNLI')
    mnli_parser.add_argument('--mnli', action='store_true', help='Run MNLI baseline')
    mnli_parser.add_argument(
        '--mnli-ckpt',
        nargs='*',
        type=str,
        default=['roberta-large-mnli']
    )

    ner_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - NER overlap',
        description='Please clone https://github.com/tingofurro/summac to baselines/summac.'
    )
    ner_parser.add_argument('--ner', action='store_true', help='Run NER overlap baseline')

    unieval_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - UniEval',
        description='Please clone https://github.com/maszhongming/UniEval to baselines/UniEval.'
    )
    unieval_parser.add_argument('--unieval', action='store_true', help='Run UniEval baseline')

    feqa_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - FEQA',
        description='Please clone https://github.com/esdurmus/feqa to baselines/feqa'
    )
    feqa_parser.add_argument('--feqa', action='store_true', help='Run FEQA baseline')

    questeval_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - QuestEval',
        description='Please clone https://github.com/ThomasScialom/QuestEval to baselines/QuestEval.'
    )
    questeval_parser.add_argument('--questeval', action='store_true', help='Run QuestEval baseline')

    qafacteval_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - QAFactEval',
        description='Please clone https://github.com/salesforce/QAFactEval to baselines/QAFactEval.'
    )
    qafacteval_parser.add_argument('--qafacteval', action='store_true', help='Run QAFactEval baseline')
        
    bleu_parser = parser.add_argument_group('Factual consistency evaluation baseline - BLEU')
    bleu_parser.add_argument('--bleu', action='store_true', help='Run BLEU baseline')
    bleu_parser.add_argument(
        '--bleu-ngram',
        nargs='*',
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4]
    )
    
    rouge_parser = parser.add_argument_group('Factual consistency evaluation baseline - ROUGE')
    rouge_parser.add_argument('--rouge', action='store_true', help='Run ROUGE baseline')
    rouge_parser.add_argument(
        '--rouge-type',
        nargs='*',
        type=str,
        choices=['1', '2', 'l'],
        default=['1', '2', 'l']
    )

    dae_parser = parser.add_argument_group('Factual consistency evaluation baseline - DAE')
    dae_parser.add_argument('--dae', action='store_true', help='Run DAE baseline')
    dae_parser.add_argument('--dae-ckpt', type=str)

    factcc_parser = parser.add_argument_group('Factual consistency evaluation baseline - FactCC')
    factcc_parser.add_argument('--factcc', action='store_true', help='Run FactCC baseline')
    factcc_parser.add_argument('--factcc-script', type=str)
    factcc_parser.add_argument('--factcc-test-data', type=str)
    factcc_parser.add_argument('--factcc-result-path', type=str)

    blanc_parser = parser.add_argument_group('Factual consistency evaluation baseline - BLANC')
    blanc_parser.add_argument('--blanc', action='store_true', help='Run BLANC baseline')
    blanc_parser.add_argument('--blanc-batch-size', type=int, default=64)

    summac_parser = parser.add_argument_group(
        'Factual consistency evaluation baseline - SummaC',
        description='Please clone https://github.com/tingofurro/summac to baselines/summac.'
    )
    summac_parser.add_argument('--summac', action='store_true', help='Run SummaC baseline')
    summac_parser.add_argument('--summac-type', nargs='*', type=str, choices=['conv', 'zs'], default=['conv', 'zs'])

    args = parser.parse_args()
    if args.timer:
        SAVE_AND_PRINT_TIMER = True

    def argugment_error(msg):
        parser.error(msg)

    run_benchmarks(args, argugment_error)
