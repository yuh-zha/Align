from model import MultitaskModel
from data import MultitaskDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
import torch

# max_steps & checkpoint_step are in backward steps,
# val_steps is in forward steps (really weird).
# see https://github.com/Lightning-AI/lightning/issues/12205
accumulate_grad_batches = 4
max_steps = 200000
checkpoint_step = 5000
val_steps = checkpoint_step * accumulate_grad_batches
batch_size = 64 // accumulate_grad_batches


def main():
    torch.set_float32_matmul_precision("high")
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=checkpoint_step,
        save_top_k=-1,
    )
    model = MultitaskModel(
        mrc_context_first=True,
        correct_mrc_loss_scaling=True,
    )
    data_module = MultitaskDataModule(".", batch_size=batch_size)
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        accelerator="gpu",
        devices=[7],
        precision=32,
        max_steps=max_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir="logs",
        val_check_interval=val_steps,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("final.ckpt")


if __name__ == "__main__":
    main()
