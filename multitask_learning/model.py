from typing import Any, List, Dict
from lightning import LightningModule
from transformers.models.roberta import RobertaModel, RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
)
from torch import nn
import torch
import numpy as np
from torch.optim import AdamW
from tasks import TASK_DEFINITION
from transformers import get_linear_schedule_with_warmup
import torchmetrics


class MultitaskModel(LightningModule):
    roberta: RobertaModel
    tokenizer: RobertaTokenizerFast

    def __init__(
        self,
        model_name: str = "roberta-base",
        tasks: List[Dict[str, Any]] = TASK_DEFINITION,
        tokenizer_args: Dict[str, Any] = dict(),
        mrc_context_first=False,
        correct_mrc_loss_scaling=False,
        learning_rate=3e-5,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        weight_decay=1e-2,
        warmup_updates=5000,
    ) -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.span_prediction_head = nn.Linear(self.roberta.config.hidden_size, 2)
        self.answarable_classifier = RobertaClassificationHead(
            RobertaConfig.from_pretrained(model_name, num_labels=2)
        )

        self.task_map = {}
        heads = []
        for task in tasks:
            self.task_map[task["name"]] = task
            index = len(heads)
            if task["type"] == "classification":
                task["index"] = index
                config = RobertaConfig.from_pretrained(
                    model_name, num_labels=task["num_labels"]
                )
                heads.append(RobertaClassificationHead(config))
            elif task["type"] == "regression":
                task["index"] = index
                config = RobertaConfig.from_pretrained(model_name, num_labels=1)
                heads.append(RobertaClassificationHead(config))
            elif task["type"] == "reading_comprehension":
                    pass
            else:
                raise RuntimeError(f"Unexpected type {task['type']}")

        self.classifiers = nn.ModuleList(heads)
        assert len(tasks) == len(self.task_map)

        self.tokenizer_args = tokenizer_args
        self.mrc_context_first = mrc_context_first
        self.mrc_loss_scaling = correct_mrc_loss_scaling

        self.save_hyperparameters()

    def tokenize(self, *args, **kwargs):
        tokenizer_args = {
            **self.tokenizer_args,
            "truncation": True,
            "padding": True,
            "return_tensors": "pt",
            **kwargs,
        }
        inputs = self.tokenizer(*args, **tokenizer_args)
        inputs_to_device = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs_to_device

    def process_batch(self, batch):
        text_a_list = []
        text_b_list = []
        params = []
        for sample in batch:
            kwargs = {**sample}
            task_name = kwargs.pop("task")
            task_type = self.task_map[task_name].get("type")
            index = self.task_map[task_name].get("index")
            num_labels = self.task_map[task_name].get("num_labels")

            kwargs["type"] = task_type
            if index is not None:
                kwargs["classifier_index"] = index
            if num_labels is not None:
                kwargs["num_labels"] = num_labels
            if task_type == "reading_comprehension":
                if self.mrc_context_first:
                    text_a = kwargs.pop("context")
                    text_b = kwargs.pop("question")
                else:
                    text_a = kwargs.pop("question")
                    text_b = kwargs.pop("context")
            else:
                text_a = kwargs.pop("text_a")
                text_b = kwargs.pop("text_b")

            text_a_list.append(text_a)
            text_b_list.append(text_b)
            params.append(kwargs)
        return text_a_list, text_b_list, params

    def forward(self, batch):
        text_a_list, text_b_list, params = self.process_batch(batch)
        inputs = self.tokenize(
            text_a_list,
            text_b_list,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        offset_mappings = inputs.pop("offset_mapping")
        special_tokens_masks = inputs.pop("special_tokens_mask")

        outputs = self.roberta(
            **inputs,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state

        losses = []
        output_logits = []
        for i, kwargs in enumerate(params):
            task_type = kwargs.pop("type")
            if task_type == "reading_comprehension":
                offset_mapping = offset_mappings[i]
                special_tokens_mask = special_tokens_masks[i]
                attention_mask = inputs["attention_mask"][i]
                special_tokens_before_context = 1 if self.mrc_context_first else 3

                is_context = (
                    torch.cumsum(special_tokens_mask, dim=-1)
                    == special_tokens_before_context
                )
                context_mask = attention_mask & (~special_tokens_mask) & is_context
                masked_offset_mapping = torch.where(
                    context_mask[..., None].bool(), offset_mapping, -1
                )
                kwargs["offset_mapping"] = masked_offset_mapping
                token_count = attention_mask.sum()
                kwargs["token_count"] = token_count

            handler = getattr(self, f"forward_{task_type}")
            loss, logits = handler(sequence_output[i], **kwargs)
            losses.append(loss)
            output_logits.append(logits)

        return losses, output_logits

    def forward_classification(
        self, sequence_output, classifier_index, num_labels=None, label=None
    ):
        classifier = self.classifiers[classifier_index]
        logits = classifier(sequence_output[None, ...])

        loss = None
        if label is not None:
            assert num_labels is not None
            loss = nn.functional.cross_entropy(
                logits.squeeze(),
                self.tensor(label, dtype=torch.long),
            )
            loss /= np.log(num_labels)
        return loss, logits

    def forward_regression(self, sequence_output, classifier_index, label=None):
        classifier = self.classifiers[classifier_index]
        logits = classifier(sequence_output[None, ...])

        loss = None
        if label is not None:
            loss = nn.functional.mse_loss(
                logits.squeeze(),
                self.tensor(label, dtype=torch.float),
            )
        return loss, logits

    def forward_reading_comprehension(
        self,
        sequence_output,
        offset_mapping=None,
        label_start=None,
        label_end=None,
        label_ans=None,
        token_count=None,
    ):
        logits = self.span_prediction_head(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        answerable_logits = self.answarable_classifier(sequence_output[None, ...])

        total_loss = None
        if label_start is not None and label_end is not None and label_ans is not None:
            assert offset_mapping is not None
            assert token_count is not None
            ignored_index = -100
            start_token = self.tensor(ignored_index, dtype=torch.long)
            end_token = self.tensor(ignored_index, dtype=torch.long)

            if label_ans:
                after_start = offset_mapping[:, 1] > label_start
                before_end = offset_mapping[:, 0] < label_end
                in_range = (after_start & before_end).long()
                start_token = torch.argmax(in_range)
                end_token = torch.argmax(torch.cumsum(in_range, dim=-1))

            answerable_label = self.tensor(label_ans, dtype=torch.long)

            loss_fct = nn.functional.cross_entropy
            start_loss = loss_fct(start_logits, start_token, ignore_index=ignored_index)
            end_loss = loss_fct(end_logits, end_token, ignore_index=ignored_index)
            answerable_loss = loss_fct(answerable_logits.squeeze(), answerable_label)

            if self.mrc_loss_scaling:
                total_loss = (start_loss + end_loss) / torch.log(token_count)
                total_loss += answerable_loss / np.log(2)
                total_loss = total_loss / 3
            else:
                total_loss = (start_loss + end_loss + answerable_loss) / 3
                total_loss /= np.log(2)

        return total_loss, (logits, answerable_logits)

    def training_step(self, batch, batch_idx):
        losses, _ = self(batch)
        for loss, sample in zip(losses, batch):
            task = sample["task"]
            self.log_dict({f"train_loss_{task}": loss}, batch_size=1)
        return torch.stack(losses).mean()

    def validation_step(self, batch, batch_idx):
        losses, _ = self(batch)
        for loss, sample in zip(losses, batch):
            task = sample["task"]
            self.log_dict(
                {f"validation_loss_{task}": loss}, batch_size=1, sync_dist=True
            )
        return torch.stack(losses).mean()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.adam_betas,
            eps=self.hparams.adam_eps,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler_config]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        for sample in batch:
            kwargs = {**sample}

            task_name = kwargs.pop("task")
            train_task_name = kwargs.pop("train_task", task_name)
            train_task_type = self.task_map[train_task_name]["type"]
            task_type = kwargs.pop("type", train_task_type)
            label = kwargs.pop("label")

            # use the training task's type and name
            kwargs["type"] = train_task_type
            kwargs["task"] = train_task_name

            label_dtype = torch.float if task_type == "regression" else torch.long
            label_tensor = self.tensor([label], dtype=label_dtype)

            handler = getattr(self, f"test_step_{task_type}")
            logits = handler(kwargs)

            metric_name = "test_metric_" + task_name
            if not hasattr(self, metric_name):
                if task_type == "classification":
                    num_labels = self.task_map[train_task_name]["num_labels"]
                    metric = torchmetrics.Accuracy(
                        task="multiclass", num_classes=num_labels
                    )
                elif task_type == "regression":
                    metric = torchmetrics.PearsonCorrCoef()
                elif task_type == "multiple_choice":
                    num_labels = len(sample["options"])
                    metric = torchmetrics.Accuracy(
                        task="multiclass", num_classes=num_labels
                    )
                elif task_type == "classification_with_label_map":
                    num_labels = len(sample["label_map"])
                    metric = torchmetrics.Accuracy(
                        task="multiclass", num_classes=num_labels
                    )
                else:
                    raise RuntimeError(f"Unknown task {task_type}")
                setattr(self, metric_name, metric.to(self.device))
            metric = getattr(self, metric_name)

            metric.update(logits, label_tensor)
            self.log(metric_name, metric, metric_attribute=metric_name)

    def test_step_classification(self, sample):
        _, logits = self([sample])
        return logits[0]

    def test_step_regression(self, sample):
        _, logits = self([sample])
        return logits[0].squeeze(0)

    def test_step_multiple_choice(self, sample):
        option_samples = []
        text_a = sample.pop("context")
        options = sample.pop("options")
        for option in options:
            option_samples.append(
                {
                    **sample,
                    "type": "classification",
                    "text_a": text_a,
                    "text_b": option,
                }
            )
        _, option_logits = self(option_samples)
        logits = torch.stack(option_logits, dim=1)
        normalized_logits = nn.functional.softmax(logits, dim=-1)
        assert normalized_logits.size(-1) == 2
        return normalized_logits[:, :, -1]

    def test_step_classification_with_label_map(self, sample):
        label_map = sample.pop("label_map")
        _, logits = self([sample])
        prediction_label = logits[0].squeeze(0).argmax().item()
        for target_label, original_labels in enumerate(label_map):
            if prediction_label in original_labels:
                return self.tensor([target_label], dtype=torch.long)
        raise RuntimeError(
            f"Unexpected label {prediction_label} for label map {label_map!r}"
        )

    def tensor(self, *args, **kwargs):
        return torch.tensor(*args, **kwargs, device=self.device)
