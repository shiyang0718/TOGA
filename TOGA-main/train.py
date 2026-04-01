import copy
import os
import time
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# util
from util.util import AverageMeter, caculat_grad, save_checkpoint, attact

# Seed

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

# IO helpers

def obtain_input(batch):
    """Dataset batch -> (text,audio,visual,label), all on cuda, label shape [B,1]"""
    if isinstance(batch, dict):
        text = batch['text'].cuda()
        audio = batch['audio'].cuda()
        visual = batch['visual'].cuda()
        label = batch['label'].cuda()
    else:
        text = batch[0].cuda()
        audio = batch[1].cuda()
        visual = batch[2].cuda()
        label = batch[3].cuda()

    if len(label.shape) == 1:
        label = label.view(-1, 1)

    return text, audio, visual, label


def train(train_loader, val_loader, model,
          train_logger, val_logger, train_batch_logger, tb_writer, opt,
          optimizer, scheduler):

    def cfg_get(k, default=None):
        if opt is None:
            return default
        if hasattr(opt, "get"):
            try:
                return opt.get(k, default)
            except Exception:
                pass
        return getattr(opt, k, default)

    def safe_setattr(obj, k, v):
        try:
            setattr(obj, k, v)
            return True
        except Exception:
            return False

    # epochs
    n_epochs = int(cfg_get("n_epochs", 20))
    val_freq = int(cfg_get("val_freq", 1))

    # loss
    loss_type = str(cfg_get("loss_type", "l1")).lower()
    loss_beta = float(cfg_get("loss_beta", 1.0))
    if loss_type == "l1":
        criterion = nn.L1Loss().cuda()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=loss_beta).cuda()
    elif loss_type == "mse":
        criterion = nn.MSELoss().cuda()
    else:
        raise ValueError(f"Unknown loss_type={loss_type}. Use l1 | smoothl1 | mse")

    # dispatch
    train_dict = {"OGM": train_OGM, "navie": train_navie}
    method = str(cfg_get("method", "navie"))
    if method not in train_dict:
        raise ValueError(f"Unknown method={method}. Only support: {list(train_dict.keys())}")
    print(f"Start training using method: {method}")

    # attack module
    att = attact(
        totel_epoch=n_epochs,
        batch=int(cfg_get("att_batch", 10))
    )

    # best
    best_mae_clean = float("inf")
    best_mae_missing = float("inf")
    befor = None

    # missing cfg
    use_curriculum = bool(cfg_get("curriculum_missing", False))
    missing_start_epoch = int(cfg_get("missing_start_epoch", 11))
    missing_type = str(cfg_get("missing_type", "t"))
    val_missing_r = float(cfg_get("val_missing_r", 0.5))

    # validation switches
    enable_val_miss = cfg_get("enable_val_miss", None)
    if enable_val_miss is None:
        enable_val_miss = use_curriculum
    else:
        enable_val_miss = bool(enable_val_miss)

    val_miss_start_epoch = int(cfg_get("val_miss_start_epoch", missing_start_epoch))
    val_miss_all_epochs = bool(cfg_get("val_miss_all_epochs", False))

    # optional: noise-only val curve
    enable_val_noise = bool(cfg_get("enable_val_noise", False))
    val_noise_std = float(cfg_get("val_noise_std", 0.2))
    val_noise_type = str(cfg_get("val_noise_type", "av"))

    # optional: stacked miss+noise
    enable_val_miss_noise = bool(cfg_get("enable_val_miss_noise", False))
    val_miss_noise_start_epoch = int(cfg_get("val_miss_noise_start_epoch", missing_start_epoch))

    print(f"[VAL CFG] enable_val_miss={enable_val_miss}, val_miss_start_epoch={val_miss_start_epoch}, "
          f"enable_val_noise={enable_val_noise}(std={val_noise_std},type={val_noise_type}), "
          f"enable_val_miss_noise={enable_val_miss_noise}(start={val_miss_noise_start_epoch})", flush=True)

    for epoch in range(1, n_epochs + 1):
        train_alg = train_dict[method]


        befor = train_alg(
            epoch=epoch,
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch_logger=train_logger,
            att=att,
            befor=befor,
            tb_writer=tb_writer,
            opt=opt
        )

        if scheduler is not None:
            scheduler.step()


        if (epoch % val_freq) != 0:
            continue

        epoch_end_step = epoch * len(train_loader)

        _bak_clean = {
            "att_eval": getattr(opt, "att_eval", False),
            "att_ration": getattr(opt, "att_ration", None),
        }
        safe_setattr(opt, "att_eval", False)
        safe_setattr(opt, "att_ration", 0.0)

        print(f"\n[VAL-CLEAN] epoch={epoch} (force clean)\n", flush=True)
        val_loss_clean, val_mae_clean = val_epoch(
            epoch, val_loader, model, criterion, val_logger,
            tb_writer=tb_writer, opt=opt, att=att, global_step=epoch_end_step,
            tb_prefix="val_clean"
        )

        safe_setattr(opt, "att_eval", _bak_clean["att_eval"])
        if _bak_clean["att_ration"] is not None:
            safe_setattr(opt, "att_ration", _bak_clean["att_ration"])

        # (B) noise-valid : optional
        if enable_val_noise:
            _bak_noise = {
                "att_eval": getattr(opt, "att_eval", False),
                "eval_att_num": getattr(opt, "eval_att_num", None),
                "eval_att_type": getattr(opt, "eval_att_type", None),
                "eval_att_ration": getattr(opt, "eval_att_ration", None),
            }
            safe_setattr(opt, "att_eval", True)
            safe_setattr(opt, "eval_att_num", "gaussian")
            safe_setattr(opt, "eval_att_type", val_noise_type)
            safe_setattr(opt, "eval_att_ration", val_noise_std)

            print(f"\n[VAL-NOISE] epoch={epoch} (gaussian {val_noise_type}, std={val_noise_std})\n", flush=True)
            _ = val_epoch(
                epoch, val_loader, model, criterion, val_logger,
                tb_writer=tb_writer, opt=opt, att=att, global_step=epoch_end_step,
                tb_prefix="val_noise"
            )

            safe_setattr(opt, "att_eval", _bak_noise["att_eval"])
            for k in ["eval_att_num", "eval_att_type", "eval_att_ration"]:
                if _bak_noise[k] is not None:
                    safe_setattr(opt, k, _bak_noise[k])

        do_val_miss = False
        if enable_val_miss:
            if val_miss_all_epochs:
                do_val_miss = True
            else:
                do_val_miss = (epoch >= val_miss_start_epoch)

        val_mae_miss = None
        if do_val_miss:
            _bak = {
                "att_eval": getattr(opt, "att_eval", False),
                "eval_att_type": getattr(opt, "eval_att_type", None),
                "eval_att_num": getattr(opt, "eval_att_num", None),
                "eval_att_ration": getattr(opt, "eval_att_ration", None),
            }

            safe_setattr(opt, "att_eval", True)
            safe_setattr(opt, "eval_att_type", missing_type)
            safe_setattr(opt, "eval_att_num", "miss")
            safe_setattr(opt, "eval_att_ration", val_missing_r)

            print(f"\n[VAL-MISS] epoch={epoch} (miss {missing_type}, r={val_missing_r})\n", flush=True)
            _, val_mae_miss = val_epoch(
                epoch, val_loader, model, criterion, val_logger,
                tb_writer=tb_writer, opt=opt, att=att, global_step=epoch_end_step,
                tb_prefix="val_miss"
            )

            safe_setattr(opt, "att_eval", _bak["att_eval"])
            for k in ["eval_att_type", "eval_att_num", "eval_att_ration"]:
                if _bak[k] is not None:
                    safe_setattr(opt, k, _bak[k])
        else:
            print(f"[VAL-MISS] skipped at epoch={epoch}", flush=True)

        do_val_miss_noise = enable_val_miss_noise and (epoch >= val_miss_noise_start_epoch)
        if do_val_miss_noise:
            _bak2 = {
                "att_eval": getattr(opt, "att_eval", False),
                "eval_stack": getattr(opt, "eval_stack", False),

                "eval_noise_num": getattr(opt, "eval_noise_num", None),
                "eval_noise_type": getattr(opt, "eval_noise_type", None),
                "eval_noise_ration": getattr(opt, "eval_noise_ration", None),

                "eval_att_num": getattr(opt, "eval_att_num", None),
                "eval_att_type": getattr(opt, "eval_att_type", None),
                "eval_att_ration": getattr(opt, "eval_att_ration", None),
            }

            safe_setattr(opt, "att_eval", True)
            safe_setattr(opt, "eval_stack", True)

            safe_setattr(opt, "eval_noise_num", "gaussian")
            safe_setattr(opt, "eval_noise_type", val_noise_type)
            safe_setattr(opt, "eval_noise_ration", val_noise_std)

            safe_setattr(opt, "eval_att_num", "miss")
            safe_setattr(opt, "eval_att_type", missing_type)
            safe_setattr(opt, "eval_att_ration", val_missing_r)

            print(f"\n[VAL-MISS+NOISE] epoch={epoch} (gaussian {val_noise_type} std={val_noise_std} + miss {missing_type} r={val_missing_r})\n", flush=True)
            _ = val_epoch(
                epoch, val_loader, model, criterion, val_logger,
                tb_writer=tb_writer, opt=opt, att=att, global_step=epoch_end_step,
                tb_prefix="val_miss_noise"
            )

            safe_setattr(opt, "att_eval", _bak2["att_eval"])
            safe_setattr(opt, "eval_stack", _bak2["eval_stack"])
            for k in ["eval_noise_num", "eval_noise_type", "eval_noise_ration",
                      "eval_att_num", "eval_att_type", "eval_att_ration"]:
                if _bak2[k] is not None:
                    safe_setattr(opt, k, _bak2[k])


        if use_curriculum and (val_mae_miss is not None):
            if epoch >= missing_start_epoch and val_mae_miss < best_mae_missing:
                best_mae_missing = val_mae_miss
                save_file_path = os.path.join(opt.result_path, "save_best_missing.pth")
                save_checkpoint(save_file_path, epoch, model, optimizer, scheduler)
                print(f"Epoch {epoch}: [BEST_MISSING] saved with miss-val MAE {best_mae_missing:.4f} "
                      f"(val_missing_r={val_missing_r})", flush=True)

        if val_mae_clean < best_mae_clean:
            best_mae_clean = val_mae_clean
            save_file_path = os.path.join(opt.result_path, "save_best.pth")
            save_checkpoint(save_file_path, epoch, model, optimizer, scheduler)
            print(f"Epoch {epoch}: [BEST_CLEAN] saved with clean-val MAE {best_mae_clean:.4f}", flush=True)

# Naive baseline

def train_navie(epoch, data_loader, model, criterion, optimizer,
                epoch_logger, att, befor, tb_writer=None, opt=None):

    print(f'Train (Naive) at epoch {epoch}')
    losses = AverageMeter()
    model.train()

    current_lr = optimizer.param_groups[-1]['lr']
    real_model = model.module if hasattr(model, 'module') else model

    def cfg_get(k, default=None):
        if opt is None:
            return default
        if hasattr(opt, "get"):
            try:
                return opt.get(k, default)
            except Exception:
                pass
        return getattr(opt, k, default)

    def _apply_attack_once(text, audio, visual, label, num, mtype, r):
        if att is None or r is None or float(r) <= 0:
            return text, audio, visual, label

        if opt is None:
            class _Tmp: pass
            tmp = _Tmp()
            tmp.att_type = mtype
            tmp.att_num = num
            tmp.att_ration = float(r)
            tmp.att_mean = 0.0
            tmp.att_std = float(r)
            return att.forward([text, audio, visual, label], tmp, float(r))

        _bak = (getattr(opt, "att_type", None),
                getattr(opt, "att_num", None),
                getattr(opt, "att_ration", None))
        try:
            setattr(opt, "att_type", mtype)
            setattr(opt, "att_num", num)
            setattr(opt, "att_ration", float(r))
            text, audio, visual, label = att.forward([text, audio, visual, label], opt, float(r))
        finally:
            try:
                if _bak[0] is not None: setattr(opt, "att_type", _bak[0])
                if _bak[1] is not None: setattr(opt, "att_num", _bak[1])
                if _bak[2] is not None: setattr(opt, "att_ration", _bak[2])
            except Exception:
                pass

        return text, audio, visual, label

    base_att_num = str(cfg_get("att_num", "miss")).lower()
    base_att_type = str(cfg_get("att_type", "av")).lower()
    base_att_r = float(cfg_get("att_ration", 0.0))

    use_curriculum = bool(cfg_get("curriculum_missing", False))
    missing_start_epoch = int(cfg_get("missing_start_epoch", 11))
    missing_end_epoch = int(cfg_get("missing_end_epoch", 20))
    miss_type = str(cfg_get("missing_type", "t")).lower()
    miss_r_start = float(cfg_get("missing_r_start", 0.3))
    miss_r_end = float(cfg_get("missing_r_end", 0.7))
    miss_sched = str(cfg_get("missing_schedule", "linear")).lower()

    miss_train = False
    miss_r = 0.0
    if use_curriculum:
        if epoch < missing_start_epoch:
            miss_train = False
            miss_r = 0.0
        elif epoch > missing_end_epoch:
            miss_train = True
            miss_r = miss_r_end
        else:
            if missing_end_epoch == missing_start_epoch:
                progress = 1.0
            else:
                progress = (epoch - missing_start_epoch) / float(missing_end_epoch - missing_start_epoch)
            if miss_sched == "step":
                miss_r = miss_r_start if progress < 0.5 else miss_r_end
            else:
                miss_r = miss_r_start + (miss_r_end - miss_r_start) * progress
            miss_train = True

    with tqdm(total=len(data_loader), desc=f'Train-epoch-{epoch}') as pbar:
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()

            global_step = (epoch - 1) * len(data_loader) + step
            text, audio, visual, label = obtain_input(batch)

            if step == 0 and epoch == 1:
                print("text:", text.shape, "audio:", audio.shape, "visual:", visual.shape)

            # gaussian
            if base_att_num == "gaussian" and base_att_r > 0:
                if step == 0:
                    print(f"[TRAIN-NOISE] epoch={epoch} gaussian type={base_att_type} std={base_att_r}", flush=True)
                text, audio, visual, label = _apply_attack_once(
                    text, audio, visual, label, num="gaussian", mtype=base_att_type, r=base_att_r
                )

            # missing
            if use_curriculum:
                if miss_train and miss_r > 0 and att is not None:
                    if step == 0:
                        print(f"[TRAIN-MISS-CUR] epoch={epoch} miss_type={miss_type} miss_r={miss_r:.2f}", flush=True)
                    text, audio, visual, label = _apply_attack_once(
                        text, audio, visual, label, num="miss", mtype=miss_type, r=miss_r
                    )
                else:
                    if step == 0:
                        print(f"[TRAIN-MISS-CUR] epoch={epoch} OFF", flush=True)
            else:
                if base_att_num == "miss" and base_att_r > 0:
                    if step == 0:
                        print(f"[TRAIN-MISS] epoch={epoch} miss_type={base_att_type} miss_r={base_att_r:.2f}", flush=True)
                    text, audio, visual, label = _apply_attack_once(
                        text, audio, visual, label, num="miss", mtype=base_att_type, r=base_att_r
                    )

            # forward/backward
            _, _, _, out = model(text, audio, visual, label, warm_up=0)
            loss = criterion(out, label)
            loss.backward()

            if tb_writer is not None and step % 10 == 0:
                g_t_list = [p.grad.norm().item() for p in real_model.text_net.parameters() if p.grad is not None]
                if g_t_list:
                    tb_writer.add_scalar('Gradient_Norm/Text', sum(g_t_list) / len(g_t_list), global_step)
                g_a_list = [p.grad.norm().item() for p in real_model.audio_net.parameters() if p.grad is not None]
                if g_a_list:
                    tb_writer.add_scalar('Gradient_Norm/Audio', sum(g_a_list) / len(g_a_list), global_step)
                g_v_list = [p.grad.norm().item() for p in real_model.visual_net.parameters() if p.grad is not None]
                if g_v_list:
                    tb_writer.add_scalar('Gradient_Norm/Visual', sum(g_v_list) / len(g_v_list), global_step)

            optimizer.step()

            losses.update(loss.item(), text.size(0))
            pbar.update(1)
            pbar.set_postfix({'Loss': losses.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_task', loss.item(), (epoch - 1) * len(data_loader) + step)

    if epoch_logger is not None:
        epoch_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': losses.avg, 'lr': current_lr})

    return befor

# OGM

def train_OGM(epoch, data_loader, model, criterion, optimizer,
              epoch_logger, att, befor, tb_writer=None, opt=None):

    print(f'Train (OGM) at epoch {epoch}')
    losses_task = AverageMeter()
    losses_total = AverageMeter()
    current_lr = optimizer.param_groups[-1]['lr']

    def cfg_get(k, default=None):
        if opt is None:
            return default
        if hasattr(opt, "get"):
            try:
                return opt.get(k, default)
            except Exception:
                pass
        return getattr(opt, k, default)

    model.train()
    real_model = model.module if hasattr(model, 'module') else model

    use_modulation = bool(cfg_get('use_hook_modulation', cfg_get('use_modulation', True)))
    warmup_epochs = int(cfg_get('warmup_epochs', 5))

    # OGM paper baseline
    ogm_paper_baseline = bool(cfg_get("ogm_paper_baseline", False))
    ogm_alpha = float(cfg_get("ogm_alpha", 8.0))
    ogm_k_min = float(cfg_get("ogm_k_min", 0.0))
    ogm_conf_mode = str(cfg_get("ogm_conf_mode", "inv_loss")).lower()

    # SoftAnchor
    use_strong_backoff = bool(cfg_get('strong_backoff', cfg_get('use_strong_backoff', False)))
    strong_lambda = float(cfg_get('strong_lambda', 0.1))
    strong_smin = float(cfg_get('strong_smin', 0.9))
    strong_w_mode = str(cfg_get('strong_w_mode', 'softmax')).lower()   # softmax / uniform
    strong_k_mode = str(cfg_get('strong_k_mode', 'soft')).lower()      # soft / hard
    strong_curriculum_epochs = int(cfg_get('strong_curriculum_epochs', 20))
    eps = 1e-8

    # tau
    strong_tau = float(cfg_get('strong_tau', 0.3))
    strong_tau_mode = str(cfg_get('strong_tau_mode', 'fixed')).lower()
    strong_tau_min = float(cfg_get('strong_tau_min', strong_tau))
    strong_tau_max = float(cfg_get('strong_tau_max', 2.0))
    if strong_tau_min > strong_tau_max:
        strong_tau_min, strong_tau_max = strong_tau_max, strong_tau_min
    strong_tau_dir = str(cfg_get('strong_tau_dir', 'down')).lower()  # up/down
    strong_tau_alpha_p = float(cfg_get('strong_tau_alpha_p', 0.35))
    strong_tau_p_stat = str(cfg_get('strong_tau_p_stat', 'range')).lower()
    strong_tau_debug = bool(cfg_get('strong_tau_debug', True))

    max_epoch = int(cfg_get('n_epochs', 20))
    start_epoch = warmup_epochs + 1

    def _spread(x: torch.Tensor, stat: str):
        if stat in ["std", "stdev"]:
            return torch.std(x, unbiased=False)
        return torch.max(x) - torch.min(x)

    def get_tau_schedule(epoch_, max_epoch_, start_epoch_, tau_min_, tau_max_, mode="cosine", direction="down"):
        if epoch_ <= start_epoch_:
            return float(tau_min_ if direction == "up" else tau_max_)
        denom = max(1, (max_epoch_ - start_epoch_))
        progress = (epoch_ - start_epoch_) / denom
        progress = max(0.0, min(1.0, progress))
        if mode == "linear":
            base = progress
        else:
            base = 0.5 * (1.0 - math.cos(math.pi * progress))
        if direction == "up":
            tau_ = tau_min_ + (tau_max_ - tau_min_) * base
        else:
            tau_ = tau_max_ - (tau_max_ - tau_min_) * base
        return float(tau_)

    def compute_tau(epoch_, perfs_, device_):
        if strong_tau_mode == "fixed":
            tau_val = float(strong_tau)
        elif strong_tau_mode in ["cosine", "linear"]:
            tau_val = get_tau_schedule(
                epoch_=epoch_,
                max_epoch_=max_epoch,
                start_epoch_=start_epoch,
                tau_min_=strong_tau_min,
                tau_max_=strong_tau_max,
                mode=strong_tau_mode,
                direction=strong_tau_dir
            )
        elif strong_tau_mode == "adaptive_p":
            base = _spread(perfs_, strong_tau_p_stat)
            tau_t = strong_tau_alpha_p * base
            tau_val = float(torch.clamp(tau_t, min=strong_tau_min, max=strong_tau_max).item())
        else:
            tau_val = float(strong_tau)

        tau_val = max(float(tau_val), 1e-6)
        return tau_val
    # GMML-Lp

    use_gmml_lp = bool(cfg_get('use_gmml_lp', False))
    lambda_p_base = float(cfg_get('lambda_p', 5e-4))
    lambda_weak_ratio = float(cfg_get('lambda_weak_ratio', 0.1))

    lp_on_modules = cfg_get("lp_on_modules", ["text_net", "audio_net", "visual_net", "fusion_model.fxy"])
    if isinstance(lp_on_modules, str):
        lp_on_modules = [s.strip() for s in lp_on_modules.split(",") if s.strip()]

    lp_missing_only = bool(cfg_get("lp_missing_only", True))
    curriculum_missing = bool(cfg_get("curriculum_missing", False))
    missing_start_epoch = int(cfg_get("missing_start_epoch", 11))

    lp_enable_epoch = int(cfg_get("lp_enable_epoch", -1))
    if lp_enable_epoch < 1:
        if lp_missing_only and curriculum_missing:
            lp_enable_epoch = missing_start_epoch
        else:
            lp_enable_epoch = 1

    lp_ramp_epochs = int(cfg_get("lp_ramp_epochs", 3))
    lp_head_ratio = float(cfg_get("lp_head_ratio", 0.3))
    lp_debug = bool(cfg_get("lp_debug", True))

    lp_apply_to_encoders = bool(cfg_get("lp_apply_to_encoders", True))
    lp_apply_to_head = bool(cfg_get("lp_apply_to_head", True))
    lp_head_only = bool(cfg_get("lp_head_only", False))
    if lp_head_only:
        lp_apply_to_encoders = False
        lp_apply_to_head = True

    if lp_head_only:
        lp_snap_modules = ["fusion_model.fxy"]
    else:
        lp_snap_modules = list(lp_on_modules)
        if not lp_apply_to_head:
            lp_snap_modules = [m for m in lp_snap_modules if ("fusion_model" not in m and "fxy" not in m)]
        if not lp_apply_to_encoders:
            lp_snap_modules = [m for m in lp_snap_modules if (("text_net" not in m) and ("audio_net" not in m) and ("visual_net" not in m))]

    base_att_num = str(cfg_get("att_num", "gaussian")).lower()
    base_att_type = str(cfg_get("att_type", "av")).lower()
    base_att_r = float(cfg_get("att_ration", 0.0))

    def _apply_attack_once(text, audio, visual, label, num, mtype, r):
        if att is None or r is None or float(r) <= 0:
            return text, audio, visual, label
        _bak = (getattr(opt, "att_type", None),
                getattr(opt, "att_num", None),
                getattr(opt, "att_ration", None))
        try:
            setattr(opt, "att_type", mtype)
            setattr(opt, "att_num", num)
            setattr(opt, "att_ration", float(r))
            text, audio, visual, label = att.forward([text, audio, visual, label], opt, float(r))
        finally:
            try:
                if _bak[0] is not None: setattr(opt, "att_type", _bak[0])
                if _bak[1] is not None: setattr(opt, "att_num", _bak[1])
                if _bak[2] is not None: setattr(opt, "att_ration", _bak[2])
            except Exception:
                pass
        return text, audio, visual, label

    def _conf_from_unimodal_losses(l_t, l_a, l_v, mode="inv_loss"):
        """conf_m bigger => stronger"""
        eps_ = 1e-8
        L = torch.stack([l_t, l_a, l_v]).detach()
        if mode in ["exp_neg", "exp(-l)", "exp"]:
            return torch.exp(-L)
        return 1.0 / (L + eps_)

    use_perf_gate = bool(cfg_get("use_perf_gate", False))
    perf_gate_start_epoch = int(cfg_get("perf_gate_start_epoch", warmup_epochs + 1))
    perf_gate_r_th = float(cfg_get("perf_gate_r_th", 0.5))          
    perf_gate_k_floor = float(cfg_get("perf_gate_k_floor", 0.2))    
    perf_gate_use_w = bool(cfg_get("perf_gate_use_w", True))        
    perf_gate_use_k = bool(cfg_get("perf_gate_use_k", True))        
    perf_gate_apply_to = str(cfg_get("perf_gate_apply_to", "tav")).lower()  
    perf_gate_min_keep = int(cfg_get("perf_gate_min_keep", 1))
    perf_gate_debug = bool(cfg_get("perf_gate_debug", False))

    def _mask_by_apply_to(mask3: torch.Tensor):
        keep = torch.tensor([
            ('t' in perf_gate_apply_to),
            ('a' in perf_gate_apply_to),
            ('v' in perf_gate_apply_to),
        ], device=mask3.device, dtype=torch.bool)
        return mask3 & keep

    def _ensure_min_keep(mask3: torch.Tensor, score3: torch.Tensor, min_keep: int):
        """Avoid gating all modalities: ensure at least `min_keep` are NOT masked."""
        if min_keep <= 0:
            return mask3
        n_mask = int(mask3.sum().item())
        if n_mask <= 3 - min_keep:
            return mask3

        max_mask = 3 - min_keep

        idx_sorted = torch.argsort(score3, descending=True)
        unmask_idx = idx_sorted[:min_keep]
        new_mask = mask3.clone()
        new_mask[unmask_idx] = False

        while int(new_mask.sum().item()) > max_mask:
            # unmask next best
            k = int((new_mask == False).sum().item())
            if k >= 3:
                break
            new_mask[idx_sorted[k]] = False
        return new_mask

    prev_perfs = None
    prev_w = None
    prev_k = None
    sum_dw, sum_dk, cnt_d = 0.0, 0.0, 0

    track_update_mag = bool(cfg_get("track_update_mag", True))
    um_log_interval = int(cfg_get("update_mag_log_interval", 10))
    um_modules = cfg_get("update_mag_on_modules", ["text_net", "audio_net", "visual_net", "fusion_model.fxy"])
    if isinstance(um_modules, str):
        um_modules = [s.strip() for s in um_modules.split(",") if s.strip()]

    with tqdm(total=len(data_loader), desc=f'Train-epoch-{epoch}') as pbar:
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            global_step = (epoch - 1) * len(data_loader) + step

            text, audio, visual, label = obtain_input(batch)
            device = text.device

            if track_update_mag and (befor is None or len(befor) == 0):
                befor = {}
                for name, p in real_model.named_parameters():
                    if any(k in name for k in um_modules) and p.requires_grad:
                        befor[name] = p.detach().clone()


            if track_update_mag and tb_writer is not None and (step % um_log_interval == 0) and befor is not None and len(befor) > 0:
                with torch.no_grad():
                    sq = {"Text": 0.0, "Audio": 0.0, "Visual": 0.0, "Head": 0.0}
                    cnt = {"Text": 0, "Audio": 0, "Visual": 0, "Head": 0}
                    for name, p in real_model.named_parameters():
                        if name not in befor or (not p.requires_grad):
                            continue
                        d = (p.detach() - befor[name]).float()
                        ds = (d * d).sum().item()
                        n = d.numel()
                        if "text_net" in name:
                            key = "Text"
                        elif "audio_net" in name:
                            key = "Audio"
                        elif "visual_net" in name:
                            key = "Visual"
                        else:
                            key = "Head"
                        sq[key] += ds
                        cnt[key] += n
                    for key in ["Text", "Audio", "Visual", "Head"]:
                        l2 = math.sqrt(max(sq[key], 0.0))
                        rms = math.sqrt(max(sq[key] / (cnt[key] + 1e-8), 0.0))
                        tb_writer.add_scalar(f"UpdateMag/Step_L2/{key}", l2, global_step)
                        tb_writer.add_scalar(f"UpdateMag/Step_RMS/{key}", rms, global_step)


            if base_att_num == "gaussian" and base_att_r > 0:
                if step == 0:
                    print(f"[TRAIN-NOISE] epoch={epoch} gaussian type={base_att_type} std={base_att_r}", flush=True)
                text, audio, visual, label = _apply_attack_once(text, audio, visual, label, num="gaussian", mtype=base_att_type, r=base_att_r)

            use_curriculum = bool(cfg_get("curriculum_missing", False))
            missing_start_epoch = int(cfg_get("missing_start_epoch", 11))
            missing_end_epoch = int(cfg_get("missing_end_epoch", 20))
            miss_type = str(cfg_get("missing_type", "t"))
            miss_r_start = float(cfg_get("missing_r_start", 0.3))
            miss_r_end = float(cfg_get("missing_r_end", 0.7))
            miss_sched = str(cfg_get("missing_schedule", "linear")).lower()

            miss_train = False
            miss_r = 0.0
            if use_curriculum:
                if epoch < missing_start_epoch:
                    miss_train = False
                elif epoch > missing_end_epoch:
                    miss_train = True
                    miss_r = miss_r_end
                else:
                    if missing_end_epoch == missing_start_epoch:
                        progress = 1.0
                    else:
                        progress = (epoch - missing_start_epoch) / float(missing_end_epoch - missing_start_epoch)
                    if miss_sched == "step":
                        miss_r = miss_r_start if progress < 0.5 else miss_r_end
                    else:
                        miss_r = miss_r_start + (miss_r_end - miss_r_start) * progress
                    miss_train = True

            if miss_train and miss_r > 0:
                if step == 0:
                    print(f"[TRAIN-MISS-CUR] epoch={epoch} miss_type={miss_type} miss_r={miss_r:.2f}", flush=True)
                text, audio, visual, label = _apply_attack_once(text, audio, visual, label, num="miss", mtype=miss_type, r=miss_r)
            else:
                if step == 0:
                    print(f"[TRAIN-MISS-CUR] epoch={epoch} OFF", flush=True)

            warm_up = 1 if epoch <= warmup_epochs else 0
            feat_t, feat_a, feat_v, out = model(text, audio, visual, label, warm_up)
            task_loss = criterion(out, label)

            perf_mode = getattr(opt, "perf_mode", "exp")
            perf_alpha = getattr(opt, "perf_alpha", 1.0)

            try:
                _, _, perf_t, perf_a, perf_v, l_t, l_a, l_v = caculat_grad(
                    model, feat_t, feat_a, feat_v, criterion, label,
                    perf_mode=perf_mode, perf_alpha=perf_alpha
                )
            except TypeError:

                _, _, perf_t, perf_a, perf_v, l_t, l_a, l_v = caculat_grad(
                    model, feat_t, feat_a, feat_v, criterion, label
                )

            Lp = torch.tensor(0.0, device=device)

            if use_gmml_lp:
                # snapshot pool includes both UpdateMag and Lp needs
                snap_modules = sorted(set(lp_snap_modules) | set(um_modules))

                if befor is None or len(befor) == 0:
                    befor = {}
                    for name, p in real_model.named_parameters():
                        if any(k in name for k in snap_modules) and p.requires_grad:
                            befor[name] = p.detach().clone()

                lp_active = (epoch >= lp_enable_epoch)

                if lp_active:
                    if lp_ramp_epochs <= 0:
                        ramp = 1.0
                    else:
                        ramp = (epoch - lp_enable_epoch + 1) / float(lp_ramp_epochs)
                        ramp = max(0.0, min(1.0, ramp))

                    lambda_p_eff = lambda_p_base * ramp

                    losses_u = torch.stack([l_t.detach(), l_a.detach(), l_v.detach()])
                    idx_strong = int(torch.argmin(losses_u).item())

                    lam_vec = torch.tensor([lambda_p_eff, lambda_p_eff, lambda_p_eff], device=device)
                    for j in range(3):
                        if j != idx_strong:
                            lam_vec[j] = lambda_p_eff * lambda_weak_ratio

                    for name, p in real_model.named_parameters():
                        if (name not in befor) or (not p.requires_grad):
                            continue

                        if 'text_net' in name:
                            if not lp_apply_to_encoders:
                                continue
                            lam = lam_vec[0]

                        elif 'audio_net' in name:
                            if not lp_apply_to_encoders:
                                continue
                            lam = lam_vec[1]

                        elif 'visual_net' in name:
                            if not lp_apply_to_encoders:
                                continue
                            lam = lam_vec[2]

                        else:
                            if not lp_apply_to_head:
                                continue
                            lam = lam_vec[idx_strong] * lp_head_ratio

                        Lp = Lp + lam * (p - befor[name]).pow(2).sum()

                    if lp_debug and step == 0:
                        print(f"[LP] epoch={epoch} active={lp_active} ramp={ramp:.2f} "
                              f"lambda_p_eff={lambda_p_eff:.2e} Lp={Lp.item():.3e} "
                              f"idx_strong={idx_strong} enc={lp_apply_to_encoders} head={lp_apply_to_head} head_only={lp_head_only}",
                              flush=True)


            use_curriculum = bool(cfg_get("curriculum_missing", False))
            missing_start_epoch = int(cfg_get("missing_start_epoch", 11))

            do_modulate = (warm_up == 0 and use_modulation and ((not use_curriculum) or (epoch >= missing_start_epoch)))
            start_epoch_eff = max(start_epoch, missing_start_epoch) if use_curriculum else start_epoch

            if do_modulate:

                perfs_raw = torch.tensor([float(perf_t), float(perf_a), float(perf_v)], device=device, dtype=torch.float32)

                alpha_prev = float(cfg_get("perf_prev_alpha", 0.5))
                use_prev_smooth = bool(cfg_get("use_perf_prev_smooth", True))
                if use_prev_smooth and (prev_perfs is not None):
                    perfs_use = alpha_prev * prev_perfs + (1.0 - alpha_prev) * perfs_raw
                else:
                    perfs_use = perfs_raw
                prev_perfs = perfs_raw.detach()

                if ogm_paper_baseline:
                    conf = _conf_from_unimodal_losses(l_t, l_a, l_v, mode=ogm_conf_mode)
                    idx_strong = int(torch.argmax(conf).item())

                    idx_all = torch.tensor([0, 1, 2], device=device)
                    idx_other = idx_all[idx_all != idx_strong]
                    rho = conf[idx_strong] / (conf[idx_other].mean() + 1e-8)

                    if rho > 1.0:
                        k_strong = 1.0 - torch.tanh(ogm_alpha * (rho - 1.0))
                    else:
                        k_strong = torch.tensor(1.0, device=device)
                    k_strong = torch.clamp(k_strong, min=ogm_k_min, max=1.0)

                    k = torch.ones_like(conf)
                    k[idx_strong] = k_strong

                    real_model.caff_t, real_model.caff_a, real_model.caff_v = k[0], k[1], k[2]

                    if prev_k is not None:
                        dk = (k - prev_k).abs().mean().item()
                        sum_dk += dk
                        cnt_d += 1
                        if tb_writer is not None and (step % 10 == 0):
                            tb_writer.add_scalar("Stability/delta_k_meanabs", float(dk), global_step)
                    prev_k = k.detach()

                elif use_strong_backoff:
                    tau = compute_tau(epoch, perfs_use, device)

                    if strong_tau_debug and step == 0 and epoch in [start_epoch_eff, start_epoch_eff + 1, start_epoch_eff + 5, max_epoch]:
                        print(f"[TAU] mode={strong_tau_mode} tau={tau:.4f} p={perfs_use.detach().cpu().numpy()}",
                              flush=True)

                    if strong_w_mode in ["uniform", "equal"]:
                        w = torch.ones_like(perfs_use) / perfs_use.numel()
                    else:
                        w = torch.softmax(perfs_use / tau, dim=0)

                    gate_on = (use_perf_gate and (epoch >= perf_gate_start_epoch))
                    if gate_on:
                        p_anchor0 = torch.sum(w * perfs_use)
                        r = perfs_use / (p_anchor0 + eps)  # [3]
                        mask_weak = (r < perf_gate_r_th)
                        mask_weak = _mask_by_apply_to(mask_weak)
                        mask_weak = _ensure_min_keep(mask_weak, r, perf_gate_min_keep)

                        if perf_gate_use_w:
                            logits = perfs_use / tau
                            logits = logits.clone()
                            logits[mask_weak] = -1e9
                            w = torch.softmax(logits, dim=0)

                        if perf_gate_debug and step == 0:
                            print(f"[GATE] epoch={epoch} r={r.detach().cpu().numpy()} mask={mask_weak.detach().cpu().numpy()} "
                                  f"use_w={perf_gate_use_w} use_k={perf_gate_use_k} k_floor={perf_gate_k_floor}", flush=True)

                    p_anchor = torch.sum(w * perfs_use)
                    gap = 1.0 - perfs_use / (p_anchor + eps)

                    lam_eff = strong_lambda * min(1.0, epoch / max(float(strong_curriculum_epochs), 1.0))

                    w_other = (1.0 - w).clamp(min=0.0)
                    w_other = w_other / (w_other.sum() + eps)

                    s_anchor = 1.0 - lam_eff * torch.sum(w_other * gap)
                    s_anchor = torch.clamp(s_anchor, min=strong_smin, max=1.0)

                    # k
                    if strong_k_mode in ["hard", "only_strong", "strong_only"]:
                        k = torch.ones_like(perfs_use)
                        idx_s = int(torch.argmax(perfs_use).item())
                        k[idx_s] = s_anchor
                    else:
                        k = 1.0 - w * (1.0 - s_anchor)

                    if gate_on and perf_gate_use_k:
                        k_floor = torch.tensor(float(perf_gate_k_floor), device=device, dtype=k.dtype)

                        k = torch.where(mask_weak, k_floor, k)

                        k = torch.where(mask_weak, torch.clamp(k, min=0.0, max=1.0),
                                        torch.clamp(k, min=strong_smin, max=1.0))
                    else:
                        k = torch.clamp(k, min=strong_smin, max=1.0)

                    if strong_tau_debug and (step == 0) and (epoch >= start_epoch_eff):
                        print(
                            f"[SOFTANCHOR] epoch={epoch} "
                            f"p={perfs_use.detach().cpu().numpy()} "
                            f"w={w.detach().cpu().numpy()} "
                            f"k={k.detach().cpu().numpy()} "
                            f"tau={float(tau):.6f} lam_eff={lam_eff:.4f} s_anchor={float(s_anchor.detach().item()):.4f}",
                            flush=True
                        )


                    if prev_w is not None and prev_k is not None:
                        dw = (w - prev_w).abs().mean().item()
                        dk = (k - prev_k).abs().mean().item()
                        sum_dw += dw
                        sum_dk += dk
                        cnt_d += 1
                        if tb_writer is not None and (step % 10 == 0):
                            tb_writer.add_scalar("Stability/delta_w_meanabs", float(dw), global_step)
                            tb_writer.add_scalar("Stability/delta_k_meanabs", float(dk), global_step)

                    prev_w = w.detach()
                    prev_k = k.detach()

                    real_model.caff_t, real_model.caff_a, real_model.caff_v = k[0], k[1], k[2]

                else:

                    real_model.caff_t = torch.tensor(1.0, device=device)
                    real_model.caff_a = torch.tensor(1.0, device=device)
                    real_model.caff_v = torch.tensor(1.0, device=device)

            else:

                real_model.caff_t = -1
                real_model.caff_a = -1
                real_model.caff_v = -1

            total_loss = task_loss + Lp
            total_loss.backward()

            if (track_update_mag or use_gmml_lp):
                if befor is None:
                    befor = {}
                snap_modules = sorted(set(lp_snap_modules) | set(um_modules))
                for name, p in real_model.named_parameters():
                    if any(k in name for k in snap_modules) and p.requires_grad:
                        befor[name] = p.detach().clone()

            optimizer.step()

            real_model.caff_t = -1
            real_model.caff_a = -1
            real_model.caff_v = -1

            losses_task.update(task_loss.item(), text.size(0))
            losses_total.update(total_loss.item(), text.size(0))

            pbar.update(1)
            pbar.set_postfix({'L_task': losses_task.avg, 'L_total': losses_total.avg})

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss_task', task_loss.item(), global_step)
                tb_writer.add_scalar('train/loss_total', total_loss.item(), global_step)

    if tb_writer is not None and cnt_d > 0:
        tb_writer.add_scalar("Stability_epoch/delta_w_meanabs", sum_dw / max(1, cnt_d), epoch)
        tb_writer.add_scalar("Stability_epoch/delta_k_meanabs", sum_dk / max(1, cnt_d), epoch)

    if epoch_logger is not None:
        epoch_logger.log({'epoch': epoch, 'loss': losses_total.avg, 'acc': losses_task.avg, 'lr': current_lr})

    return befor


def val_epoch(epoch, data_loader, model, criterion, logger, tb_writer, opt,
              att=None, global_step=None, tb_prefix="val"):

    print(f'Validation at epoch {epoch}')
    model.eval()

    losses = AverageMeter()
    mae_meter = AverageMeter()

    all_preds = []
    all_labels = []

    att_eval = bool(getattr(opt, "att_eval", False))

    def _apply_attack_once(text, audio, visual, label, num, mtype, r):
        if (r is None) or (float(r) <= 0):
            return text, audio, visual, label

        _bak = (
            getattr(opt, "att_type", None),
            getattr(opt, "att_num", None),
            getattr(opt, "att_ration", None)
        )
        try:
            setattr(opt, "att_type", mtype)
            setattr(opt, "att_num", num)
            setattr(opt, "att_ration", float(r))
            text, audio, visual, label = att.forward([text, audio, visual, label], opt, float(r))
        finally:
            try:
                if _bak[0] is not None:
                    setattr(opt, "att_type", _bak[0])
                if _bak[1] is not None:
                    setattr(opt, "att_num", _bak[1])
                if _bak[2] is not None:
                    setattr(opt, "att_ration", _bak[2])
            except Exception:
                pass

        return text, audio, visual, label

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            text, audio, visual, label = obtain_input(batch)

            if att_eval and (att is not None):
                eval_stack = bool(getattr(opt, "eval_stack", False))

                if eval_stack:
                    noise_num = str(getattr(opt, "eval_noise_num", "gaussian"))
                    noise_type = str(getattr(opt, "eval_noise_type", "av"))
                    noise_r = float(getattr(opt, "eval_noise_ration", 0.0))

                    miss_num = str(getattr(opt, "eval_att_num", "miss"))
                    miss_type = str(getattr(opt, "eval_att_type", "t"))
                    miss_r = float(getattr(opt, "eval_att_ration", 0.0))

                    if i == 0 and (noise_r > 0 or miss_r > 0):
                        print(f"[EVAL STACK] noise=({noise_num},{noise_type},{noise_r}) + "
                              f"miss=({miss_num},{miss_type},{miss_r})", flush=True)

                    text, audio, visual, label = _apply_attack_once(text, audio, visual, label, noise_num, noise_type, noise_r)
                    text, audio, visual, label = _apply_attack_once(text, audio, visual, label, miss_num, miss_type, miss_r)

                else:
                    eval_num = str(getattr(opt, "eval_att_num", "miss"))
                    eval_type = str(getattr(opt, "eval_att_type", getattr(opt, "att_type", "t")))
                    eval_r = float(getattr(opt, "eval_att_ration", getattr(opt, "att_ration", 0.0)))

                    if i == 0 and eval_r > 0:
                        print(f"[EVAL ATT] num={eval_num} type={eval_type} r={eval_r}", flush=True)

                    text, audio, visual, label = _apply_attack_once(text, audio, visual, label, eval_num, eval_type, eval_r)

            _, _, _, out = model(text, audio, visual, label, warm_up=1)

            loss = criterion(out, label)
            mae = torch.mean(torch.abs(out - label))

            losses.update(loss.item(), text.size(0))
            mae_meter.update(mae.item(), text.size(0))

            all_preds.append(out.detach().view(-1).cpu())
            all_labels.append(label.detach().view(-1).cpu())

    corr, acc2, f1 = 0.0, 0.0, 0.0
    if len(all_preds) > 0 and len(all_labels) > 0:
        preds_np = torch.cat(all_preds).numpy()
        labels_np = torch.cat(all_labels).numpy()

        if float(np.std(preds_np)) >= 1e-8 and float(np.std(labels_np)) >= 1e-8:
            corr = float(np.corrcoef(preds_np, labels_np)[0, 1])
            if np.isnan(corr) or np.isinf(corr):
                corr = 0.0

        pred_pos = preds_np >= 0
        label_pos = labels_np >= 0
        acc2 = float(np.mean(pred_pos == label_pos))

        tp = int(np.sum(pred_pos & label_pos))
        fp = int(np.sum(pred_pos & (~label_pos)))
        fn = int(np.sum((~pred_pos) & label_pos))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = float(2 * prec * rec / (prec + rec + 1e-8))

    gs = int(global_step) if global_step is not None else int(epoch)

    print(
        f"Epoch: [{epoch}]\t Loss: {losses.avg:.4f}\t MAE: {mae_meter.avg:.4f}"
        f"\t Corr: {corr:.4f}\t Acc-2: {acc2*100:.2f}%\t F1: {f1:.4f}"
    )

    if logger is not None:
        logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': mae_meter.avg,
            'acc_num': 0,
            'mae': mae_meter.avg,
            'corr': corr,
            'acc2': acc2,
            'f1': f1,
        })

    if tb_writer is not None:
        tb_writer.add_scalar(f'{tb_prefix}/loss', losses.avg, gs)
        tb_writer.add_scalar(f'{tb_prefix}/mae', mae_meter.avg, gs)
        tb_writer.add_scalar(f'{tb_prefix}/corr', corr, gs)
        tb_writer.add_scalar(f'{tb_prefix}/acc2', acc2, gs)
        tb_writer.add_scalar(f'{tb_prefix}/f1', f1, gs)

    return losses.avg, mae_meter.avg
