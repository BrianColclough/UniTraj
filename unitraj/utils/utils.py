import glob
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# How lane tokens may exchange information. 'stock' is the upstream model, which has no lane-to-lane attention.
CONDITIONS = ('stock', 'learned', 'structured', 'random')


def find_latest_checkpoint(search_pattern):

    # List all files matching the pattern
    list_of_files = glob.glob(search_pattern, recursive=True)
    # Find the file with the latest modification time
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch Lightning based training.

    Args:
    seed_value (int): The seed value to be set for random number generators.
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed_value)

    # If using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    # Set the random seed for numpy (if using numpy in the project)
    np.random.seed(seed_value)

    # Set the random seed for Python's `random`
    random.seed(seed_value)

    # Set the seed for PyTorch Lightning's internal operations
    pl.seed_everything(seed_value, workers=True)


def compose_config(overrides=()):
    """
    The config train.py would get for these command line overrides, for use outside of a hydra app.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name='config', overrides=list(overrides))
    OmegaConf.set_struct(cfg, False)
    return OmegaConf.merge(cfg, cfg.method)


def seed_worker(worker_id):
    """
    Seed numpy and random inside a dataloader worker.

    PyTorch gives every worker its own torch seed (base_seed + worker_id, with base_seed drawn from the
    loader's generator) but leaves numpy and random with whatever state the worker was started with.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch_size, num_workers, seed, shuffle=False, collate_fn=None):
    """
    Dataloader whose shuffling order and worker random states are a function of the seed only.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=False,
        collate_fn=collate_fn, worker_init_fn=seed_worker, generator=generator)


def build_loggers(cfg):
    """
    Loggers listed in cfg.logger, all writing below cfg.output_dir. csv and tensorboard need no account.
    """
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
    loggers = []
    for name in cfg.logger:
        if name == 'csv':
            loggers.append(CSVLogger(save_dir=cfg.output_dir, name='csv'))
        elif name == 'tensorboard':
            loggers.append(TensorBoardLogger(save_dir=cfg.output_dir, name='tensorboard'))
        elif name == 'wandb':
            os.makedirs(cfg.output_dir, exist_ok=True)
            loggers.append(WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name, save_dir=cfg.output_dir))
        else:
            raise ValueError(f'Unknown logger {name}, expected csv, tensorboard or wandb')
    return loggers


def check_condition(cfg):
    if cfg.condition not in CONDITIONS:
        raise ValueError(f'Unknown condition {cfg.condition}, expected one of {CONDITIONS}')
    if cfg.condition != 'stock':
        # TODO: build the lane-to-lane attention mask for this condition
        print(f'Warning: condition={cfg.condition} is not implemented yet, '
              f'this run uses the stock model without lane-to-lane attention')
