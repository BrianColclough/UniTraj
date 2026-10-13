import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, build_dataloader, build_loggers, check_condition
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    check_condition(cfg)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    ckpt_dir = os.path.join(cfg.output_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.ckpt_monitor,  # Replace with your validation metric
        # the metric name contains a slash, which would otherwise end up as a directory in the file name
        filename='epoch={epoch}-' + cfg.ckpt_monitor.split('/')[-1] + '={' + cfg.ckpt_monitor + ':.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=ckpt_dir
    )

    call_backs.append(checkpoint_callback)

    train_loader = build_dataloader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, seed=cfg.seed,
        shuffle=cfg.shuffle_train, collate_fn=train_set.collate_fn)

    val_loader = build_dataloader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, seed=cfg.seed,
        shuffle=False, collate_fn=train_set.collate_fn)

    # debug only switches off logging and multi-gpu training, it still runs on the gpu if there is one
    use_gpu = torch.cuda.is_available()
    multi_gpu = use_gpu and not cfg.debug and len(cfg.devices) > 1
    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else build_loggers(cfg),
        devices=cfg.devices if use_gpu and not cfg.debug else 1,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="gpu" if use_gpu else "cpu",
        profiler=cfg.profiler,
        strategy="ddp" if multi_gpu else "auto",
        callbacks=call_backs
    )

    # automatically resume training
    if cfg.ckpt_path is None and cfg.auto_resume and not cfg.debug:
        last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            cfg.ckpt_path = last_ckpt

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
