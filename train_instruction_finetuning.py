from pytorch_lightning import Trainer, seed_everything
from align.dataloader import FLANDataLoader
from align.model import InstructFinetuningModel
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import os


def train(datasets, args):
    dm = FLANDataLoader(
        dataset_config=datasets,
        model_name=args.model_name,
        sample_mode='seq',
        train_batch_size=args.batch_size,
        eval_batch_size=16, 
        num_workers=args.num_workers,
        train_eval_split=0.95,
    )
    dm.setup()

    model = InstructFinetuningModel(
        model=args.model_name, 
        adam_epsilon=args.adam_epsilon,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps_portion=args.warm_up_proportion
    )

    checkpoint_name = '_'.join((
        f"{args.ckpt_comment}{args.model_name.replace('/', '-')}",
        str(args.max_samples_per_dataset),
        f"{args.batch_size}x{len(args.devices)}x{args.accumulate_grad_batch}"
    ))
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_save_path,
        filename=checkpoint_name + "_{epoch:02d}_{step}",
        every_n_train_steps=10000,
        save_top_k=1
    )
    trainer = Trainer(
        accelerator='gpu', 
        max_steps=args.max_steps, 
        devices=args.devices, 
        strategy="dp", 
        precision=32,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batch
    )

    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(os.path.join(args.ckpt_save_path, f"{checkpoint_name}_final.ckpt"))

    print("Training is finished.")
    
if __name__ == "__main__":
    FLAN_TRAINING_DATASETS = {
        ### NLI
        'mnli': {'flan':'mnli', 'task_type': 'nli', 'data_path': 'mnli.json'},     
        'doc_nli': {'flan':'rte', 'task_type': 'bin_nli', 'data_path': 'doc_nli.json'},
        'snli': {'flan':'snli', 'task_type': 'nli', 'data_path': 'snli.json'},
        'anli_r1': {'flan':'anli', 'task_type': 'nli', 'data_path': 'anli_r1.json'},
        'anli_r2': {'flan':'anli', 'task_type': 'nli', 'data_path': 'anli_r2.json'},
        'anli_r3': {'flan':'anli', 'task_type': 'nli', 'data_path': 'anli_r3.json'},

        ### fact checking
        'nli_fever': {'flan':'mnli', 'task_type': 'fact_checking', 'data_path': 'nli_fever.json'},
        'vitaminc': {'flan':'mnli', 'task_type': 'fact_checking', 'data_path': 'vitaminc.json'},

        ### paraphrase
        'paws': {'flan':'paws_wiki', 'task_type': 'paraphrase', 'data_path': 'paws.json'},
        'paws_qqp': {'flan':'paws_wiki', 'task_type': 'paraphrase', 'data_path': 'paws_qqp.json'},
        'paws_unlabeled': {'flan':'paws_wiki', 'task_type': 'paraphrase', 'data_path': 'paws_unlabeled.json'},
        'qqp': {'flan':'glue_qqp', 'task_type': 'paraphrase', 'data_path': 'qqp.json'},
        'wiki103': {'flan':'paws_wiki', 'task_type': 'paraphrase', 'data_path': 'wiki103.json'},

        ### QA
        'squad_v2': {'flan':'squad_v2', 'task_type': 'qa', 'data_path': 'squad_v2.json'},
        'race': {'flan':'openbookqa', 'task_type': 'qa', 'data_path': 'race.json'},
        'adversarial_qa': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'adversarial_qa.json'},
        'drop': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'drop.json'},
        'hotpot_qa_distractor': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'hotpot_qa_distractor.json'},
        'hotpot_qa_fullwiki': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'hotpot_qa_fullwiki.json'},
        'newsqa': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'newsqa.json'},
        'quoref': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'quoref.json'},
        'ropes': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'ropes.json'},
        'boolq': {'flan':'bool_q', 'task_type': 'qa', 'data_path': 'boolq.json'},
        'quail': {'flan':'openbookqa', 'task_type': 'qa', 'data_path': 'quail.json'},
        'sciq': {'flan':'squad_v1', 'task_type': 'qa', 'data_path': 'sciq.json'},
        'strategy_qa': {'flan':'bool_q', 'task_type': 'qa', 'data_path': 'strategy_qa.json'},

        ### Coreference
        'gap': {'flan':'gap', 'task_type': 'coreference', 'data_path': 'gap.json'},

        ### Summarization
        'wikihow': {'flan':'xsum', 'task_type': 'summarization', 'data_path': 'wikihow.json'},

        ### Information Retrieval
        'msmarco': {'flan':'msmarco', 'task_type': 'ir', 'data_path': 'msmarco.json'},

        ### STS
        'stsb': {'flan':'stsb', 'task_type': 'sts', 'data_path': 'stsb.json'},
        'sick': {'flan':'stsb', 'task_type': 'sts', 'data_path': 'sick.json'},
    }

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulate-grad-batch', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=84000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--warm-up-proportion', type=float, default=0.06)
    parser.add_argument('--adam-epsilon', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--val-check-interval', type=float, default=1. / 4)
    parser.add_argument('--devices', nargs='+', type=int, required=True)
    parser.add_argument('--model-name', type=str, default="t5-base")
    parser.add_argument('--ckpt-save-path', type=str, required=True)
    parser.add_argument('--ckpt-comment', type=str, default="")
    parser.add_argument('--trainin-datasets', nargs='+', type=str, default=list(FLAN_TRAINING_DATASETS.keys()), choices=list(FLAN_TRAINING_DATASETS.keys()))
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--max-samples-per-dataset', type=int, default=500000)
   
    args = parser.parse_args()
    
    seed_everything(args.seed)

    datasets = {
        name: {
            **FLAN_TRAINING_DATASETS[name],
            "size": args.max_samples_per_dataset,
            "data_path": os.path.join(args.data_path, FLAN_TRAINING_DATASETS[name]['data_path'])
        }
        for name in args.trainin_datasets
    }
    
    train(datasets, args)


