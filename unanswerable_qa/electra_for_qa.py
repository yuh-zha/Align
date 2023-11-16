from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AdamW,
    ElectraModel,
    ElectraTokenizerFast,
    get_linear_schedule_with_warmup
)
from typing import Any, List, Tuple, Optional, Dict
import torch
from torch.nn import functional, CrossEntropyLoss, Linear, GELU, Sequential
import re
from torch.utils.data import DataLoader, Dataset, default_collate
import datasets
from collections import defaultdict
import numpy as np
import random
from . import squad_eval
import json


def argmax_right(x: torch.Tensor) -> torch.LongTensor:
    f = x.flip(-1)
    return x.size(-1) - 1 - f.argmax(dim=-1)


def get_squad_score(reference_answers, prediction):
    gold_answers = [a for a in reference_answers
                    if squad_eval.normalize_answer(a)]
    if not gold_answers:
        # For unanswerable questions, only correct answer is empty string
        gold_answers = ['']

    # Take max over all gold answers
    exact_score = max(squad_eval.compute_exact(a, prediction)
                      for a in gold_answers)
    f1_score = max(squad_eval.compute_f1(a, prediction)
                   for a in gold_answers)
    return exact_score, f1_score


class ElectraForQA(LightningModule):
    def __init__(self, model='google/electra-large-discriminator',
                 learning_rate=5e-5,
                 adam_epsilon=1e-6,
                 adam_weight_decay=0.01,
                 adam_betas=(0.9, 0.999),
                 layerwise_lr_decay=0.9,
                 warmup_frac=0.1,
                 answerable_weight=0.5,
                 beam_size=20,
                 **kwargs) -> None:
        super().__init__()
        self.model = ElectraModel.from_pretrained(model)
        self.tokenizer = ElectraTokenizerFast.from_pretrained(model)

        hidden_size = self.model.config.hidden_size

        self.answer_start_cls = Linear(hidden_size, 1)
        # Slightly different from
        # https://github.com/google-research/electra/blob/8a46635f32083ada044d7e9ad09604742600ee7b/finetune/qa/qa_tasks.py#L513
        # where they used an output dimension of 512 instead of hidden_size.
        self.answer_end_cls = Sequential(
            Linear(hidden_size * 2, hidden_size),  # <---
            GELU(),
            Linear(hidden_size, 1)
        )
        self.answerable_cls = Sequential(
            Linear(hidden_size * 2, hidden_size),  # <---
            GELU(),
            Linear(hidden_size, 1)
        )

        self.save_hyperparameters()

    def beam_search(self, end_logits,
                    contexts, context_mask,
                    start_top_log_p, start_top_index,
                    offset_mapping,
                    ) -> List[List[str]]:
        # Enforce end index >= start index
        index = torch.arange(end_logits.size(-1), device=self.device)
        end_token_mask = index[None, None, :] >= start_top_index[..., None]
        end_token_mask = end_token_mask.long()
        masked_end_logits = end_logits + 1000.0 * (end_token_mask - 1)

        end_log_p = functional.log_softmax(masked_end_logits, dim=-1)
        end_top_log_p, end_top_index = torch.topk(
            end_log_p,
            k=min(self.hparams.beam_size, end_log_p.size(-1))
        )

        start_end_log_p = start_top_log_p.unsqueeze(-1) + end_top_log_p
        beams = start_end_log_p.view(start_end_log_p.size(0), -1)

        beam_top_log_p, beam_top_index = torch.topk(
            beams,
            k=min(self.hparams.beam_size, beams.size(-1))
        )
        beam_top_index = beam_top_index.cpu().numpy()

        # assert beams.size(-1) == self.hparams.beam_size ** 2
        # index into start_top_index & end_top_index
        start_i, end_j = np.unravel_index(
            beam_top_index,
            start_end_log_p.shape[1:]
        )

        start_top_index = start_top_index.cpu()
        start_top_log_p = start_top_log_p.cpu()
        end_top_index = end_top_index.cpu()
        end_top_log_p = end_top_log_p.cpu()
        beam_top_log_p = beam_top_log_p.cpu()
        context_mask = context_mask.cpu()
        offset_mapping = offset_mapping.cpu()

        all_answers = []
        for s_index, s_log_p, e_index, e_log_p, mask, offsets, context in \
                zip(
                    start_top_index,
                    start_top_log_p,
                    end_top_index,
                    end_top_log_p,
                    context_mask,
                    offset_mapping,
                    contexts
                ):
            sample_answers = []
            for start, sp, ends, eps in zip(s_index, s_log_p, e_index, e_log_p):
                beam_answers = []
                for end, ep in zip(ends, eps):
                    if start > end:
                        continue
                    if not mask[start] or not mask[end]:
                        continue
                    char_start = offsets[start, 0].item()
                    char_end = offsets[end, 1].item()
                    beam_answers.append(
                        (context[char_start:char_end], ep.item()))
                sample_answers.append((sp.item(), beam_answers))
            all_answers.append(sample_answers)

        top_answers = []
        for i, j, s_index, e_index, batch_log_p, mask, offsets, context in \
                zip(
                    start_i,
                    end_j,
                    start_top_index,
                    end_top_index,
                    beam_top_log_p,
                    context_mask,
                    offset_mapping,
                    contexts
                ):
            answers = []
            for start, end, log_p in zip(s_index[i], e_index[i, j], batch_log_p):
                if start > end:
                    continue
                if not mask[start] or not mask[end]:
                    continue
                char_start = offsets[start, 0].item()
                char_end = offsets[end, 1].item()
                answers.append((context[char_start:char_end], log_p.item()))
            top_answers.append(answers)
        return top_answers, all_answers

    def forward(self,
                questions: List[str], contexts: List[str],
                answer_spans: Optional[List[Tuple[int, int]]] = None) -> Any:

        inputs = self.tokenizer(
            questions, contexts,
            padding=True,
            truncation='only_second',
            return_tensors='pt',
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )

        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        offset_mapping = model_inputs.pop('offset_mapping')
        special_tokens_mask = model_inputs.pop('special_tokens_mask')

        context_mask = model_inputs['attention_mask'] * \
            model_inputs['token_type_ids'] * (1 - special_tokens_mask)

        labels = None
        if answer_spans is not None:
            span_start, span_end = answer_spans
            answerable_labels = span_start < span_end

            # answer_token_mask is 1 iff the token is a part of the answer
            answer_token_mask = context_mask \
                * (offset_mapping[:, :, 1] > span_start.unsqueeze(-1))\
                * (offset_mapping[:, :, 0] < span_end.unsqueeze(-1))

            # find start & end positions
            start_positions = answer_token_mask.argmax(dim=-1)
            end_positions = argmax_right(answer_token_mask)

            # if no answer, set start & end to ignore_index
            ignored_index = context_mask.size(-1)
            no_answer = ~answerable_labels
            start_positions[no_answer] = ignored_index
            end_positions[no_answer] = ignored_index

            labels = {
                'start_positions': start_positions,
                'end_positions': end_positions,
                'ignored_index': ignored_index,
                'answerable_labels': answerable_labels
            }

        outputs = self.model(**model_inputs)
        output_dict = dict()

        last_hidden_state = outputs.last_hidden_state
        pooled_feature = last_hidden_state[:, 0]

        # Predict start positions
        start_logits = self.answer_start_cls(last_hidden_state).squeeze(-1)
        start_logits += 1000.0 * (context_mask - 1)
        output_dict['start_logits'] = start_logits

        # Predict end positions
        start_log_p = functional.log_softmax(start_logits, dim=-1)
        start_top_log_p, start_top_index = torch.topk(
            start_log_p,
            k=min(self.hparams.beam_size, start_log_p.size(-1))
        )

        if self.training:
            assert labels is not None and 'start_positions' in labels
            # When training, use the ground truth start_positions
            # We use values >= sequence length to denote a ignored
            # label, but for the purpose of gathering start token
            # embeddings we need to clamp that to < sequence length.
            start_index = labels['start_positions']\
                .unsqueeze(-1)\
                .clamp(0, context_mask.size(-1) - 1)
        else:
            start_index = start_top_index

        # last_hidden_state (batch, length, hidden size)
        # start_index     (batch, beams)
        # start_embedding   (batch, beams, hidden size)
        # end_feature       (batch, beams, length, hidden size * 2)
        start_index_expanded = start_index\
            .unsqueeze(-1).expand(-1, -1, last_hidden_state.size(-1))
        start_embedding = last_hidden_state.gather(-2, start_index_expanded)
        end_feature = torch.cat(
            torch.broadcast_tensors(
                last_hidden_state.unsqueeze(1),
                start_embedding.unsqueeze(2)
            ),
            dim=-1
        )
        end_logits = self.answer_end_cls(end_feature).squeeze(-1)
        end_logits += 1000.0 * (context_mask - 1).unsqueeze(1)
        output_dict['end_logits'] = end_logits

        # Predict answerable labels
        # https://github.com/google-research/electra/blob/8a46635f32083ada044d7e9ad09604742600ee7b/finetune/qa/qa_tasks.py#L507-L515
        start_p = functional.softmax(start_logits, dim=-1)
        start_feature = (start_p.unsqueeze(-1) * last_hidden_state).sum(dim=1)
        answerable_feature = torch.cat((pooled_feature, start_feature), dim=-1)
        answerable_logits = self.answerable_cls(answerable_feature).squeeze(-1)
        output_dict['answerable_logits'] = answerable_logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=labels['ignored_index'])
            start_loss = loss_fct(start_logits, labels['start_positions'])
            # We use the 0th beam to calculate end_loss.
            # This is prediction conditioned on either the ground truth
            # start positions (training) or the top predicted start positions (eval).
            end_loss = loss_fct(end_logits[:, 0], labels['end_positions'])
            answerable_loss = functional.binary_cross_entropy_with_logits(
                answerable_logits, answerable_labels.float()
            )

            total_loss = (start_loss + end_loss) / 2 + \
                answerable_loss * self.hparams.answerable_weight

            # If all samples in the batch are not answerable, the loss will be NaN.
            # In that case we ask the optimizer to optimize the answerable_loss only.
            loss = total_loss
            if torch.isnan(loss):
                assert not labels['answerable_labels'].any()
                loss = answerable_loss

            output_dict['loss'] = total_loss
            output_dict['start_loss'] = start_loss
            output_dict['end_loss'] = end_loss
            output_dict['answerable_loss'] = answerable_loss

        # Do beam search to find the best answers
        if not self.training:
            top_answers, all_answers = self.beam_search(
                end_logits=end_logits,
                contexts=contexts,
                context_mask=context_mask,
                start_top_log_p=start_top_log_p,
                start_top_index=start_top_index,
                offset_mapping=offset_mapping,
            )
            output_dict['predicted_answers'] = top_answers
            output_dict['all_predicted_answers'] = all_answers

        return loss, output_dict

    def configure_optimizers(self):

        pattern = re.compile(r'(?!\W)encoder\.layer\.(\d+)\.')

        def get_layer_num(param_name):
            m = re.search(pattern, param_name)
            if m:
                return int(m.group(1))
            return None

        def get_encoder_layer_count(param_names):
            layers = set()
            for name in param_names:
                layer_num = get_layer_num(name)
                if layer_num is not None:
                    layers.add(layer_num)
            return len(layers)

        param_names = (name for name, _ in self.named_parameters())
        total_encoder_layers = get_encoder_layer_count(param_names)

        def get_layer_depth(param_name):
            items = param_name.split('.')
            if 'encoder' in items:
                layer_num = get_layer_num(param_name)
                assert layer_num is not None
                return total_encoder_layers - get_layer_num(param_name)
            elif 'embeddings' in items:
                return total_encoder_layers + 1
            else:
                return 0

        def use_weight_decay(param_name):
            items = param_name.split('.')
            return all(n not in items for n in ["bias", "LayerNorm"])

        param_group = defaultdict(list)
        for name, param in self.named_parameters():
            depth = get_layer_depth(name)
            decay = use_weight_decay(name)
            param_group[(depth, decay)].append(param)

        optimizer_grouped_parameters = [
            {
                "params": params,
                "lr": self.hparams.learning_rate * (self.hparams.layerwise_lr_decay ** depth),
                "weight_decay": self.hparams.adam_weight_decay if decay else 0.
            }
            for (depth, decay), params in param_group.items()
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=self.hparams.adam_betas,
            eps=self.hparams.adam_epsilon,
        )
        num_warmup_steps = int(
            self.hparams.warmup_frac *
            self.trainer.estimated_stepping_batches
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        all_answers = batch.pop('all_answers')

        loss, outputs = self(**batch)

        log_dict = {
            f'train_{k}': v
            for k, v in outputs.items()
            if 'loss' in k and not torch.isnan(v)
        }

        self.log_dict(
            log_dict,
            on_epoch=True,
            on_step=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        all_answers = batch.pop('all_answers')

        loss, outputs = self(**batch)

        answerable_p = torch.sigmoid(outputs['answerable_logits'])

        predicted_answers = [
            greedy_pick_candidate_answer(candidates, p)
            for candidates, p in zip(outputs['predicted_answers'], answerable_p)
        ]

        exact_match, f1 = zip(*(
            get_squad_score(references, predictions)
            for references, predictions
            in zip(all_answers, predicted_answers)
        ))

        log_dict = {
            f'val_{k}': v
            for k, v in outputs.items()
            if 'loss' in k and not torch.isnan(v)
        }

        log_dict['val_exact_match'] = np.mean(exact_match)
        log_dict['val_f1'] = np.mean(f1)

        self.log_dict(
            log_dict,
            on_epoch=True,
            on_step=True,
            logger=True
        )

        return loss

    def predict_step(self, batch, batch_idx):
        sample_ids = batch.pop('id')

        _, outputs = self(**batch)

        return {
            **batch,
            'sample_ids': sample_ids,
            'predictions': outputs['predicted_answers'],
            'all_predictions': outputs['all_predicted_answers'],
            'answerable_logits': outputs['answerable_logits'].cpu().tolist()
        }


def greedy_pick_candidate_answer(candidates, answerable_p, **kwargs):
    if answerable_p > 0.5:
        return candidates[0]
    return ''


def collate_fn(batch):
    all_answers = [sample.pop('all_answers') for sample in batch]
    collated = default_collate(batch)
    collated['all_answers'] = all_answers
    return collated


class SQuAD2Dataset(Dataset):
    def __init__(self, split) -> None:
        super().__init__()
        self.split = split

    def __getitem__(self, index):
        sample = self.split[index]

        answer_spans = (0, 0)
        answers = sample['answers']
        num_answers = min(len(answers['text']), len(answers['answer_start']))
        if num_answers > 0:
            index = random.randrange(num_answers)
            start = answers['answer_start'][index]
            end = start + len(answers['text'][index])
            answer_spans = (start, end)

        return {
            'questions': sample['question'],
            'contexts': sample['context'],
            'answer_spans': answer_spans,
            'all_answers': answers['text']
        }

    def __len__(self):
        return len(self.split)


class SQuAD2JsonDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        with open(path, 'r', encoding='utf8') as file:
            raw_data = json.load(file)
        self.data = []
        for document in raw_data['data']:
            for paragraph in document['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    self.data.append({
                        'contexts': context,
                        'questions': qa['question'],
                        'id': qa['id'],
                    })

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SQuAD2PredictionDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.data = dataset

    def __getitem__(self, index):
        sample = self.data[index]
        return {
            'contexts': sample['context'],
            'questions': sample['question'],
            'id': sample['id'],
        }

    def __len__(self):
        return len(self.data)


class SQuAD2DataLoader(LightningDataModule):
    def __init__(self, dataset='squad_v2', batch_size=16):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = datasets.load_dataset(self.hparams.dataset)
        self.train_dataset = SQuAD2Dataset(dataset['train'])
        self.validation_dataset = SQuAD2Dataset(dataset['validation'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            shuffle=True, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, collate_fn=collate_fn
        )


if __name__ == '__main__':
    seed_everything(42)

    max_epochs = 3

    model = ElectraForQA(max_epochs=max_epochs, learning_rate=1e-5)
    datamodule = SQuAD2DataLoader(batch_size=8)

    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch',
        patience=10,
        mode='min'
    )
    loss_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_epoch',
        filename='{epoch}-{val_loss_epoch:.6f}',
        save_top_k=10,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        accelerator='gpu',
        devices=[4],
        callbacks=[
            early_stop_callback,
            loss_checkpoint_callback,
            lr_monitor
        ],
        default_root_dir='electra_squad2',
        max_epochs=max_epochs,
        val_check_interval=0.2
    )

    trainer.fit(model, datamodule=datamodule)
