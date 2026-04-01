import json
import random
import os
import numpy as np
import torch
import torch.nn as nn
import hydra
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from utils import get_dataset
from util.util import attact 
from models.multimodal import Classifier
from train import train, val_epoch, setup_seed
# from train_baselines import train
from utils import * 

def _build_loaders_fallback(cfg):

    bs = int(getattr(cfg, "batch_size", 32))
    nw = int(getattr(cfg, "n_threads", 0))

    val_split = "valid"
    try:
        val_split = str(getattr(cfg.dataset, "eval_split", "valid"))
    except Exception:
        pass

    def _inst(split: str):

        last_err = None
        for kwargs in (
            {"mode": split},
            {"mode": "train", "eval_split": split},
            {"eval_split": split},
        ):
            try:
                return hydra.utils.instantiate(cfg.dataset, **kwargs)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to instantiate dataset for split='{split}': {last_err}")

    train_set = _inst("train")
    val_set = _inst(val_split)

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg):
    model = Classifier(cfg)
    return model


@hydra.main(config_path='cfgs', config_name='train', version_base=None)
def main(cfg):
    print(f"Current Configuration: {cfg}")
    print("[INFO] main.py is using train_baselines.train for training.", flush=True)

    if hasattr(cfg, 'random_seed'):
        setup_seed(cfg.random_seed)
    else:
        setup_seed(42)

    model = build_model(cfg)
    print("Model Structure Built.")

    if cfg.train:

        if getattr(cfg, "tensorboard", False):
            if not os.path.exists(cfg.result_path):
                os.makedirs(cfg.result_path)
            tb_writer = SummaryWriter(log_dir=cfg.result_path)
        else:
            tb_writer = None

        (train_loader, val_loader) = get_dataset(cfg)

        if train_loader is None:
            print('[WARN] utils.get_dataset returned train_loader=None; using fallback DataLoader builder.')
            train_loader, val_loader = _build_loaders_fallback(cfg)
        if val_loader is None:
            print('[WARN] utils.get_dataset returned val_loader=None; using fallback DataLoader builder.')
            _, val_loader = _build_loaders_fallback(cfg)

        try:
            train_n = len(train_loader.dataset) if train_loader is not None else -1
            val_n = len(val_loader.dataset) if val_loader is not None else -1
            print(f"[DATA] train_size={train_n} | val_size={val_n} | batch_size={getattr(cfg,'batch_size', 'NA')}")
            if train_n == val_n and train_n > 0:
                print("[WARN] train_size == val_size. Please confirm the dataset split logic (train/valid may be the same split!).")
        except Exception as e:
            print(f"[DATA] Could not print dataset sizes: {e}")

        train_logger, train_batch_logger, val_logger = get_logger(cfg)

        if hasattr(cfg, 'gpu_device') and len(cfg.gpu_device) > 1:
            torch.distributed.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        else:
            model = model.cuda()

        parameters = [p for p in model.parameters() if p.requires_grad]

        try:
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
        except Exception as e:
            print(f"Hydra optimizer instantiation failed: {e}. Fallback to manual AdamW.")
            optimizer = torch.optim.AdamW(parameters, lr=getattr(cfg, "lr", 1e-4), weight_decay=1e-4)

        try:
            scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        except Exception:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        train(train_loader, val_loader, model,
              train_logger, val_logger, train_batch_logger, tb_writer, cfg,
              optimizer, scheduler)

    else:

        assert hasattr(cfg, "ckpt_path") and cfg.ckpt_path, "Please set cfg.ckpt_path to a checkpoint file."

        cfg.val = True
        _, val_loader = get_dataset(cfg)

        if val_loader is None:
            print('[WARN] utils.get_dataset returned val_loader=None in eval-only; using fallback DataLoader builder.')
            _, val_loader = _build_loaders_fallback(cfg)

        need_att = bool(getattr(cfg, "att_eval", False)) or bool(getattr(cfg, "att_train", False))
        att = attact(totel_epoch=cfg.n_epochs, batch=10) if need_att else None

        if getattr(cfg, "loss_type", "l1") == "smoothl1":
            criterion = nn.SmoothL1Loss(beta=getattr(cfg, "loss_beta", 1.0)).cuda()
        else:
            criterion = nn.L1Loss().cuda()

        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model = model.cuda()

        val_epoch(epoch=0, data_loader=val_loader, model=model,
                  criterion=criterion, logger=None, tb_writer=None, opt=cfg, att=att)


if __name__ == '__main__':
    main()
