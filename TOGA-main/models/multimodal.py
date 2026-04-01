import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import csv
from .fusion_model import ConcatFusion
from .backbone import mosi_encoder


class custom_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, theta):
        ctx.save_for_backward(input, theta)
        if 1 - theta.item() == 0:
            return input
        return input / (1 - theta.item())

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        if 1 - theta.item() == 0:
            return grad_output, theta
        input_grad = 1 / (1 - theta.item()) * grad_output.clone()
        return input_grad, theta


class Modality_drop():
    def __init__(self, dim_list, p_exe=0.7, device='cuda'):
        self.dim_list = dim_list
        self.p_exe = p_exe
        self.device = device

    def execute_drop(self, fead_list, q):

        B = fead_list[0].shape[0]
        D = fead_list[0].shape[1]

        exe_drop = torch.tensor(np.random.rand(1)).to(device=self.device) >= 1 - self.p_exe
        if not exe_drop:
            return fead_list, torch.ones([B], dtype=torch.int32, device=self.device)
        num_mod = len(fead_list)
        theta = torch.mean(q)
        mask = torch.distributions.Bernoulli(1 - q).sample([B, 1]).permute(2, 1, 0).contiguous().reshape(num_mod, B,
                                                                                                         -1).to(
            device=self.device)

        concat_list = torch.stack(fead_list, dim=0)  # [3, B, D]
        concat_list = torch.mul(concat_list, mask)
        concat_list = custom_autograd.apply(concat_list, theta)

        mask = torch.transpose(mask, 0, 1).squeeze(-1)  # [B, 3]
        update_flag = torch.sum(mask, dim=1) > 0
        cleaned_fea = torch.chunk(concat_list, num_mod, dim=0)
        cleaned_fea = [_.squeeze(0) for _ in cleaned_fea]  # list of [B, D]

        return cleaned_fea, update_flag


class Classifier(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device

        ds_cfg = getattr(cfg, "dataset", None)
        dim_t = int(getattr(ds_cfg, "input_d_t", getattr(cfg, "input_d_t", 768))) if ds_cfg is not None else int(
            getattr(cfg, "input_d_t", 768))
        dim_a = int(getattr(ds_cfg, "input_d_a", getattr(cfg, "input_d_a", 5))) if ds_cfg is not None else int(
            getattr(cfg, "input_d_a", 5))
        dim_v = int(getattr(ds_cfg, "input_d_v", getattr(cfg, "input_d_v", 20))) if ds_cfg is not None else int(
            getattr(cfg, "input_d_v", 20))
        self.map_dim = 128

        self.text_net = mosi_encoder(dim_t, hidden_dim=128, output_dim=self.map_dim)
        self.audio_net = mosi_encoder(dim_a, hidden_dim=32, output_dim=self.map_dim)
        self.visual_net = mosi_encoder(dim_v, hidden_dim=64, output_dim=self.map_dim)

        self.norm_w = None

        self.fusion_model = ConcatFusion(dim_t=self.map_dim, dim_a=self.map_dim, dim_v=self.map_dim, out_c=1)
        self.register_hook()

        if hasattr(self.cfg, 'use_adam_drop') and self.cfg.use_adam_drop:
            self.modality_drop = Modality_drop(dim_list=[self.map_dim] * 3, p_exe=self.cfg.p_exe, device=self.device)

        self.bz_history = []

        self.caff_t = -1
        self.caff_a = -1
        self.caff_v = -1

    def register_hook(self):
        def caculate_deta_graid(grad, device: str):

            if self.caff_t != -1 and self.caff_a != -1 and self.caff_v != -1:

                c_t = self.caff_t if isinstance(self.caff_t, float) else self.caff_t.item()
                c_a = self.caff_a if isinstance(self.caff_a, float) else self.caff_a.item()
                c_v = self.caff_v if isinstance(self.caff_v, float) else self.caff_v.item()
                dim = self.map_dim  # 128

                grad[:, :dim] = grad[:, :dim] * c_t + \
                                torch.zeros_like(grad[:, :dim]).normal_(0, 1e-4)

                grad[:, dim:2 * dim] = grad[:, dim:2 * dim] * c_a + \
                                       torch.zeros_like(grad[:, dim:2 * dim]).normal_(0, 1e-4)

                grad[:, 2 * dim:] = grad[:, 2 * dim:] * c_v + \
                                    torch.zeros_like(grad[:, 2 * dim:]).normal_(0, 1e-4)

            return (grad.to(device),)

        def backward_hook(module, grad_input, grad_output):
            if len(grad_input) <= 0 or grad_input[0] is None:
                return grad_input

            g0 = grad_input[0]

            modified_g0 = caculate_deta_graid(g0.to('cpu'), g0.device)[0]
            new_grad_input = list(grad_input)
            new_grad_input[0] = modified_g0
            return tuple(new_grad_input)

        for name, module in self.named_modules():
            if 'fxy' in name:
                module.register_full_backward_hook(backward_hook)

    def forward(self, text, audio, vision, label=None, warm_up=1):

        feat_t = self.text_net(text)
        feat_a = self.audio_net(audio)
        feat_v = self.visual_net(vision)

        if hasattr(self, 'modality_drop') and warm_up == 0:
            pass

        t_out, a_out, v_out, out = self.fusion_model(feat_t, feat_a, feat_v)

        return feat_t, feat_a, feat_v, out
