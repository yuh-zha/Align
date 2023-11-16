from model import MultitaskModel
from data import get_test_dataset, get_unseen_test_dataset, _no_op
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything
import torch
import json
import os

batch_size = 64
ckpt = "logs/lightning_logs/version_2/checkpoints/epoch=2-step=200000.ckpt"


def main():
    torch.set_float32_matmul_precision("high")
    seed_everything(42)

    model = MultitaskModel.load_from_checkpoint(ckpt, map_location='cpu')
    seen_dataloader = DataLoader(
        get_test_dataset(),
        batch_size=64,
        collate_fn=_no_op,
    )
    unseen_dataloader = DataLoader(
        get_unseen_test_dataset(),
        batch_size=64,
        collate_fn=_no_op,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=[4],
        precision=32,
        default_root_dir="test_logs",
    )

    results = trainer.test(model, [seen_dataloader, unseen_dataloader])
    with open(os.path.join(trainer.log_dir, "test_results.json"), "w") as file:
        json.dump({"ckpt": ckpt, "results": results}, file)


if __name__ == "__main__":
    main()
