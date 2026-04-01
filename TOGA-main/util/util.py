# import copy
# import time
# import torch
# import torch.nn as nn
# import os
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import torch.nn.functional as F
# import random
# import torch
# from torchvision.transforms import transforms, functional
# class attact():
#
#
#     def __init__(self,totel_epoch,batch,epoch_list=None):
#
#         self.epoch = None
#         if epoch_list!=None:
#             self.epoch=epoch_list
#         else:
#             self.epoch = random.randint(1, totel_epoch)  # Randomly generate an epoch between 1 and 100
#         if isinstance(batch, list):
#             self.batch = batch
#         else:
#             self.batch = [random.randint(0, batch) for _ in range(batch)]
#
#     def __create_mask(self,data,ration):
#         mask = torch.zeros(data.shape[:1], dtype=torch.bool)
#
#         index = torch.randperm(data.shape[:1][0])[:int(data.shape[:1][0]*ration)]
#         mask[index] = True
#
#         return mask
#     def __make_track(self,HW:list,R:float):
#         H = HW[0]
#         W = HW[1]
#         # Create a tensor of ones with the same height (H) and width (W) as the input
#         mask = torch.ones(H, W)
#
#         # Calculate the dimensions of the rectangle to be masked
#         rect_width = int(W * R)
#         rect_height = int(H * R)
#
#         # Randomly select the top-left corner of the rectangle within the bounds of the mask
#         start_x = random.randint(0, W - rect_width - 1)
#         start_y = random.randint(0, H - rect_height - 1)
#
#         # Set the selected rectangle region to 0
#         mask[start_y:start_y + rect_height, start_x:start_x + rect_width] = 0
#         # print(mask[start_y:start_y + rect_height, start_x:start_x + rect_width])
#         # print(rect_width,rect_height)
#         return mask
#         # print(x,y)
#         # print(h,w)
#
#     def __make_pattern(self,inputdata,pattern_tensor):
#         # inputdata：[B,C,W,H]
#         max_value = -10
#         full_image = torch.zeros(inputdata.shape)
#         full_image.fill_(max_value)
#
#
#         x =3
#         y = 23
#         x_bot = x+ pattern_tensor.shape[0]
#
#         y_bot = y+ pattern_tensor.shape[1]
#
#         full_image[...,x:x_bot,y:y_bot] = pattern_tensor
#
#
#         if full_image.shape[-3] ==3:
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#         else:
#             normalize = transforms.Normalize(mean=[0.485],
#                                      std=[0.229])
#
#         mask = 1*(full_image!=max_value).to(inputdata.device)
#         full_image = normalize(full_image).to(inputdata.device)
#         # print(full_image[mask==1])
#         # print(inputdata.device,full_image.device,mask.device)
#         inputdata=(1-mask) * inputdata + mask*full_image
#
#         return inputdata
#
#
#
#     def random_Gaussian(self,data,ration,ty:str,mean=0, std=1):
#         orgin_audio = data[0]
#         orgin_vision = data[1]
#         orgin_label= data[2]
#         mask = self.__create_mask(orgin_audio, ration)
#         if 'a' in ty:
#             audio_noise = orgin_audio+torch.randn_like(orgin_audio).to(orgin_audio.device) * std + mean
#             orgin_audio[mask] = audio_noise[mask]
#             print('gaussian in audio')
#         if 'v' in ty:
#             vision_noise = orgin_vision +  torch.randn_like(orgin_vision).to(orgin_audio.device) * std + mean
#             orgin_vision[mask] = vision_noise[mask]
#             print('gaussian in vision'),
#         return orgin_audio, orgin_vision,orgin_label
#     def get(self):
#         return self.epoch,self.batch
#     def trick(self,data,ration,ty:str,R=0.1):
#         orgin_audio = data[0]
#         orgin_vision = data[1]
#         orgin_label= data[2]
#         mask = self.__create_mask(orgin_audio, ration)
#         if 'a' in ty:
#             trick_audio = orgin_audio[mask]
#             # print(trick_audio.shape[-1],trick_audio.shape[-2])
#             W = trick_audio.shape[-1]
#             H = trick_audio.shape[-2]
#             trick_a = self.__make_track([H,W],R)
#             # print(trick_audio.shape,trick.shape)
#             print('trick in audio')
#             orgin_audio[mask] = trick_audio*trick_a.to(orgin_audio.device)
#         if 'v' in ty:
#             trick_vision = orgin_vision[mask]
#             W_v = trick_vision.shape[-1]
#             H_v = trick_vision.shape[-2]
#             trick_v = self.__make_track([H_v,W_v],R)
#             # print(trick_v.shape)
#             print('trick in vision')
#             orgin_vision[mask] = orgin_vision[mask]*trick_v.to(orgin_vision.device)
#         return orgin_audio, orgin_vision,orgin_label
#     def random_label(self,data,ration):
#         orgin_audio = data[0]
#         orgin_vision = data[1]
#         orgin_label= data[2]
#         mask = self.__create_mask(orgin_label,ration)
#         operation_label =  orgin_label[mask]
#         random_temp  = torch.randint(0,31,operation_label.shape).to(orgin_audio.device)
#         for i in range(len(operation_label)):
#             while random_temp[i] == operation_label[i]:
#                 random_temp[i] = torch.randint(0,31,(1,)).item()
#         orgin_label[mask] = random_temp
#
#
#     def one_pixel_attack(self,data,ration,ty,test):
#         orgin_audio = data[0]
#         orgin_vision = data[1]
#         orgin_label= data[2]
#         pattern_tensor = torch.tensor([[1.]])
#         mask = self.__create_mask(orgin_label,ration)
#         if ty ==  'a':
#             operation_audio =  orgin_audio[mask]
#             operation_audio = self.__make_pattern(operation_audio,pattern_tensor)
#             orgin_audio[mask] = operation_audio.to(orgin_audio.device)
#         if ty =='v':
#             operation_vision =  orgin_vision[mask]
#             operation_vision = self.__make_pattern(operation_vision,pattern_tensor)
#             orgin_vision[mask]  = operation_vision.to(orgin_audio.device)
#         # print(orgin_label[mask])
#         if test==False:
#             orgin_label[mask] = orgin_label[mask].fill_(1)
#
#         # print(orgin_label)
#         return orgin_audio, orgin_vision,orgin_label
#
#     def nine_pixel_attack(self,data,ration,ty,test):
#         orgin_audio = data[0]
#         orgin_vision = data[1]
#         orgin_label= data[2]
#         pattern_tensor = torch.tensor([
#                                         [1., 0., 1.],
#                                         [-10., 1., -10.],
#                                         [-10., -10., 0.],
#                                         [-10., 1., -10.],
#                                         [1., 0., 1.]
#                                     ])
#         mask = self.__create_mask(orgin_label,ration)
#         if ty ==  'a':
#             operation_audio =  orgin_audio[mask]
#             operation_audio = self.__make_pattern(operation_audio,pattern_tensor)
#             orgin_audio[mask] = operation_audio.to(orgin_audio.device)
#         if ty =='v':
#             operation_vision =  orgin_vision[mask]
#             operation_vision = self.__make_pattern(operation_vision,pattern_tensor)
#             orgin_vision[mask]  = operation_vision.to(orgin_audio.device)
#         # print(orgin_label[mask])
#         if test ==False:
#             orgin_label[mask] = orgin_label[mask].fill_(1)
#
#         # print(orgin_label)
#         return orgin_audio, orgin_vision,orgin_label
#     def miss_modal(self,data,ration,ty):
#         orgin_audio = data[0]
#         orgin_vision = data[1]
#         orgin_label= data[2]
#         mask = self.__create_mask(orgin_audio, ration)
#         if 'a' in ty:
#             orgin_audio[mask] = 0
#             print('miss in audio')
#         if 'v' in ty:
#             orgin_vision[mask] = 0
#             print('miss in vision')
#         return orgin_audio, orgin_vision,orgin_label
#     def forward(self, data, opt,ration,R=0.1):
#
#             # print('opt.att_type:',opt.att_type)
#             ty = opt.att_type
#
#             if opt.att_num != None:
#                 attack_type = opt.att_num
#             if attack_type == 'gaussian':
#
#                 return self.random_Gaussian(data,ration,ty,mean=opt.att_mean,std=opt.att_std)
#             elif attack_type == 'trick':
#                 data = self.random_label(data,ration)
#                 return self.trick(data, ration,ty,R)
#             elif attack_type=='one':
#                 # print('oneonenoen')
#                 data = self.one_pixel_attack(data,ration,ty,opt.att_test)
#                 return data
#             elif attack_type == 'nine':
#                 data = self.nine_pixel_attack(data,ration,ty,opt.att_test)
#                 return data
#             elif attack_type == 'miss':
#                 data = self.miss_modal(data,ration,ty)
#                 return data
#             else:
#                 data = self.random_Gaussian(data, ration,ty,mean=opt.mean,std=opt.std)
#                 data = self.trick(data, ration, ty,R)
#                 return data
#
#
# class TQ():
#     def __init__(self, max_length):
#         self.max_length = max_length
#         self.queue = []
#
#     def enqueue(self, tensor):
#         if len(self.queue) >= self.max_length:
#             self.queue.pop(0)  # 出队
#         self.queue.append(tensor)  # 入队
#
#     def get_queue(self):
#         return torch.stack(self.queue) if self.queue else torch.tensor([])
#
#     def get_average(self):
#         if not self.queue:
#             return torch.tensor([])
#         return torch.mean(torch.stack(self.queue), dim=0)
#     def get_queue_length(self):
#         return len(self.queue)
#     def get_first(self):
#         return self.queue[0]
# class Hessian():
#     def __init__(self, param_dict):
#         self.grad_queue = {key: TQ(2) for key in param_dict.keys()}
#         for key in param_dict.keys():
#             self.grad_queue[key].enqueue(param_dict[key])
#     def get_hessian(self,param_dict,lr):
#
#         hessian = {key:0 for key in param_dict.keys()}
#         for key in param_dict.keys():
#             if param_dict[key] is not None and self.grad_queue[key].get_queue_length() == 2:
#                 hessian_temp = param_dict[key] + self.grad_queue[key].get_first()
#                 hessian[key] =  torch.sum(hessian_temp).item() / lr
#             elif  self.grad_queue[key].get_queue_length() != 2:
#                 print('queue length is not enough')
#             self.grad_queue[key].enqueue(param_dict[key])
#
#         return hessian
#
# import torch
#
# def caculat_grad(model,out_1,out_2,criterion,label,Val=False):
#     parameters = copy.deepcopy(dict(model.named_parameters())) # 复制模型参数
#     cos_sim = 0
#     grad_v,grad_a,grad_c=0. ,0. ,0.
#     # print(parameters.keys())
#     performance_1=0
#     performance_2=0
#     # 复制输入，开启梯度追踪
#     out_1 = copy.deepcopy(out_1.detach()).double().requires_grad_(True)
#     out_2 = copy.deepcopy(out_2.detach()).double().requires_grad_(True)
#
#     # 手动提取最后的全连接层权重
#     W = parameters['module.fusion_model.fxy.weight'].double()
#     b = (parameters['module.fusion_model.fxy.bias'].double()/2).detach().requires_grad_(True)
#     B = (parameters['module.fusion_model.fxy.bias'].double()).detach().requires_grad_(True)
#     W_1 = W[:,:512].detach().requires_grad_(True)
#
#     W_2 = W[:,512:].detach().requires_grad_(True)
#     w = W.detach().requires_grad_(True)
#     # print(W_1)
#     # 手动计算单模态预测结果
#     t_a = F.linear(out_1, W_1 ,b)
#     t_v = F.linear(out_2, W_2,b)
#     t_f = F.linear(torch.cat([out_1,out_2],dim=1),w,B)
#     # print('t1',t1.shape)
#
#     # 计算单模态loss
#     l_a = criterion(t_a, label)
#     l_v = criterion(t_v, label)
#     l_f = criterion(t_f,label)
#     # 计算梯度
#
#     if Val == True:
#         acc1 = sum([torch.argmax(t_a,dim=1)[i] == label[i].item() for i in range(t_a.shape[0])])
#         acc2 = sum([torch.argmax(t_v,dim=1)[i] == label[i].item() for i in range(t_v.shape[0])])
#         return acc1,acc2
#     grad_a = list(torch.autograd.grad(l_a, [out_1, W_1, b ], create_graph=True, allow_unused=True))
#     grad_v = list(torch.autograd.grad(l_v, [out_2, W_2, b ], create_graph=True, allow_unused=True))
#     grad_f = list(torch.autograd.grad(l_f, [torch.cat([out_1,out_2]), w, B], create_graph=True, allow_unused=True))
#
#     cos_sim=calculate_cosine_similarity(grad_a[1], grad_v[1])
#     grad_cf = torch.cat((grad_a[1],grad_v[1]),dim=1)
#     # print(torch.sum(grad_cf),torch.sum(grad_f[1]))
#
#     # print(grad_cf.shape,grad_f[1].shape)
#     param_dict = {'audio':W_1,'visual':W_2,'fusion':w,'cf_f_sim':calculate_cosine_similarity(grad_f[1], grad_cf)}
#     l_a.backward()
#     l_v.backward()
#     l_f.backward()
#     # 计算单模态performance：取出真实标签对应的那个softmax概率值，累加
#     performance_1=sum([F.softmax(t_a,dim=1)[i][int(label[i].item())] for i in range(t_a.shape[0])])
#     performance_2=sum([F.softmax(t_v,dim=1)[i][int(label[i].item())] for i in range(t_v.shape[0])])
#     # print(torch.sum(grad_a[1]),torch.sum(grad_v[1]),torch.sum(grad_f[1]))
#     print(torch.argmax(t_a,dim=1)[0],label[0])
#
#     return cos_sim,[grad_a,grad_v,grad_f,param_dict],performance_1.float(),performance_2.float(),l_a,l_v
#
# def calculate_cosine_similarity(grad1, grad2):
#     """
#     计算两个梯度的余弦相似度。
#
#     Args:
#         grad1 (torch.Tensor): 第一个梯度张量。
#         grad2 (torch.Tensor): 第二个梯度张量。
#
#     Returns:
#         float: 余弦相似度的平均值。
#     """
#     if isinstance(grad1, tuple):
#         grad1 = torch.tensor(grad1)
#     if isinstance(grad2, tuple):
#         grad2 = torch.tensor(grad2)
#     grad1_flat = grad1.reshape(-1)
#     grad2_flat = grad2.reshape(-1)
#     # print(grad1_flat.shape)
#     if grad1_flat.size(0) != grad2_flat.size(0):
#         return 0
#         # raise ValueError('The size of grad1 and grad2 must be the same.')
#
#     cos_sim = F.cosine_similarity(grad1_flat, grad2_flat,dim=0)
#     # print(cos_sim)
#     return cos_sim.item()
# def EMA(befor,now,t, alpha):
#     """
#     计算指数移动平均 (EMA)
#
#     参数:
#     data (list): 时间序列数据
#     alpha (float): 平滑因子，0 < alpha <= 1
#
#     返回:
#     list: EMA 值列表
#
#     使用demo:
#     for index,number in enumerate(arr):
#     if index==0:
#         print(number)
#     else:
#         print(exponential_moving_average(arr[index-1],number,0.2))
#         arr[index]=exponential_moving_average(arr[index-1],number,0.2)
#     """
#
#     return ((1-alpha)*now+(alpha)*befor)
#
# def comperssor(num,ema:list,threshold=0.5,ratio=100):
#
#     if abs(num-ema[1]) >threshold:
#         print('go this')
#         return torch.sign(num) * (ema[0] + (abs(num) - ema[0]) / ratio)
#     else:
#         return num
#
#
# def c_process(input_num,history_num,step,EMA_hyper=0.99,T=0.5,R=100):
#     input_norm = input_num.norm()
#     if type(history_num) == float:
#         history_norm =input_norm
#     else:
#         history_norm = history_num.norm()
#     if step==0:
#         ema = input_norm
#         comper_norm = input_norm
#     else:
#
#         ema = EMA(history_norm,input_norm,step,EMA_hyper)
#         comper_norm = comperssor(history_norm,[history_norm,ema],T,R)
#     if (comper_norm/input_norm) <1:
#         comper_ema = input_num * (comper_norm/input_norm)
#     print((comper_norm/input_norm))
#     return comper_ema

import copy
import time
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import random
import math

# 【注意】移除了 torchvision 依赖，因为 MOSI 是特征向量，不用图像变换

class attact():
    def __init__(self, totel_epoch, batch, epoch_list=None):
        self.epoch = None
        if epoch_list != None:
            self.epoch = epoch_list
        else:
            self.epoch = random.randint(1, totel_epoch)

        if isinstance(batch, list):
            self.batch = batch
        else:
            self.batch = [random.randint(0, batch) for _ in range(batch)]

    def __create_mask(self, data, ration):
        mask = torch.zeros(data.shape[:1], dtype=torch.bool)
        # 随机选择一部分样本进行攻击/增强
        index = torch.randperm(data.shape[:1][0])[:int(data.shape[:1][0] * ration)]
        mask[index] = True
        return mask

    def random_Gaussian(self, data, ration, ty: str, mean=0.0, std=0.2):
        """
        Gaussian for feature inputs (MOSI/MOSEI):
        - std: 噪声强度
        - ration: 在你的新设定里不再用于mask比例（保留接口兼容即可）
        """
        text, audio, visual, label = data

        # 注意：ty 是 att_type，比如 't'/'a'/'v'/'av'/'tav'
        ty = str(ty).lower()

        if 't' in ty:
            text = text + torch.randn_like(text) * std + mean

        if 'a' in ty:
            audio = audio + torch.randn_like(audio) * std + mean

        if 'v' in ty:
            visual = visual + torch.randn_like(visual) * std + mean

        return text, audio, visual, label

    # def random_Gaussian(self, data, ration, ty: str, mean=0, std=1):
    #     # data: [text, audio, visual, label]
    #     origin_text = data[0]
    #     origin_audio = data[1]
    #     origin_visual = data[2]
    #     origin_label = data[3]
    #
    #     # 只根据 Audio 的 batch size 生成 mask，其他模态共用
    #     mask = self.__create_mask(origin_audio, ration)
    #
    #     # 给 Text 加噪声 (如果有需求)
    #     if 't' in ty:
    #         text_noise = origin_text + torch.randn_like(origin_text).to(origin_text.device) * std + mean
    #         origin_text[mask] = text_noise[mask]
    #
    #     # 给 Audio 加噪声
    #     if 'a' in ty:
    #         audio_noise = origin_audio + torch.randn_like(origin_audio).to(origin_audio.device) * std + mean
    #         origin_audio[mask] = audio_noise[mask]
    #
    #     # 给 Visual 加噪声
    #     if 'v' in ty:
    #         visual_noise = origin_visual + torch.randn_like(origin_visual).to(origin_visual.device) * std + mean
    #         origin_visual[mask] = visual_noise[mask]
    #
    #     return origin_text, origin_audio, origin_visual, origin_label

    # def miss_modal(self, data, ration, ty):
    #     # 模态缺失模拟：直接置零
    #     origin_text = data[0]
    #     origin_audio = data[1]
    #     origin_visual = data[2]
    #     origin_label = data[3]
    #
    #     mask = self.__create_mask(origin_audio, ration)
    #
    #     if 't' in ty:
    #         origin_text[mask] = 0
    #         # print('miss in text')
    #     if 'a' in ty:
    #         origin_audio[mask] = 0
    #         # print('miss in audio')
    #     if 'v' in ty:
    #         origin_visual[mask] = 0
    #         # print('miss in visual')
    #
    #     return origin_text, origin_audio, origin_visual, origin_label
    def miss_modal(self, data, ration, ty):
        # 模态缺失模拟：sample-level，把整条样本的某个模态置零
        text, audio, visual, label = data
        device = text.device
        B = text.size(0)

        # m: [B]，表示该样本是否 missing（1=missing）
        m = (torch.rand(B, device=device) < ration).float()

        def apply_missing(x, m_1d):
            # 把 [B] reshape 成 [B,1,1,...] 以便 broadcast 到任意维度
            view = [B] + [1] * (x.dim() - 1)
            m_view = m_1d.view(*view)
            return x * (1.0 - m_view)  # missing -> 0

        # 注意：这里默认同一批样本在 ty 指定的模态上“同步缺失”
        if 't' in ty:
            text = apply_missing(text, m)
        if 'a' in ty:
            audio = apply_missing(audio, m)
        if 'v' in ty:
            visual = apply_missing(visual, m)

        return text, audio, visual, label

    def get(self):
        return self.epoch, self.batch

    # def forward(self, data, opt, ration, R=0.1):
    #     # 统一入口
    #     ty = opt.att_type
    #     if hasattr(opt, 'att_num') and opt.att_num is not None:
    #         attack_type = opt.att_num
    #     else:
    #         attack_type = 'gaussian'  # 默认
    #
    #     # if attack_type == 'gaussian':
    #     #     return self.random_Gaussian(data, ration, ty, mean=opt.att_mean, std=opt.att_std)
    #     # util/util.py  inside class attact, def forward(...)
    #     if attack_type == 'gaussian':
    #         mean = float(getattr(opt, "att_mean", 0.0))
    #         #关键：如果配置里没有 att_std，则把 ration 当作 std 用（你命令行传的就是 0.1/0.2/0.3）
    #         std = float(getattr(opt, "att_std", ration))
    #         return self.random_Gaussian(data, ration, ty, mean=mean, std=std)
    #     elif attack_type == 'miss':
    #         return self.miss_modal(data, ration, ty)
    #     else:
    #         # 默认 fallback 到高斯噪声
    #         return self.random_Gaussian(data, ration, ty, mean=0, std=0.1)
    def forward(self, data, opt, ration, R=0.1):
        ty = getattr(opt, "att_type", "av")
        attack_type = getattr(opt, "att_num", "gaussian")

        if attack_type == "gaussian":
            mean = float(getattr(opt, "att_mean", 0.0))
            std = float(getattr(opt, "att_std", ration))

            ty = str(opt.att_type).lower()
            if ty in ["rand1", "random1", "single_random"]:
                ty = random.choice(["t", "a", "v"])  # ✅每个 batch 随机选一个模态

            return self.random_Gaussian(data, ration, ty, mean=mean, std=std)

        elif attack_type == "miss":
            return self.miss_modal(data, ration, ty)

        else:
            return data


# TQ, Hessian 类保留 (辅助优化器逻辑)
class TQ():
    def __init__(self, max_length):
        self.max_length = max_length
        self.queue = []

    def enqueue(self, tensor):
        if len(self.queue) >= self.max_length:
            self.queue.pop(0)
        self.queue.append(tensor)

    def get_queue(self):
        return torch.stack(self.queue) if self.queue else torch.tensor([])

    def get_average(self):
        if not self.queue:
            return torch.tensor([])
        return torch.mean(torch.stack(self.queue), dim=0)

    def get_queue_length(self):
        return len(self.queue)

    def get_first(self):
        return self.queue[0]


class Hessian():
    def __init__(self, param_dict):
        self.grad_queue = {key: TQ(2) for key in param_dict.keys()}
        for key in param_dict.keys():
            self.grad_queue[key].enqueue(param_dict[key])

    def get_hessian(self, param_dict, lr):
        hessian = {key: 0 for key in param_dict.keys()}
        for key in param_dict.keys():
            if param_dict[key] is not None and self.grad_queue[key].get_queue_length() == 2:
                hessian_temp = param_dict[key] + self.grad_queue[key].get_first()
                hessian[key] = torch.sum(hessian_temp).item() / lr
            self.grad_queue[key].enqueue(param_dict[key])
        return hessian


# 【核心修改】适应 MOSI 的 3 模态梯度计算
def caculat_grad(model, feat_t, feat_a, feat_v, criterion, label, Val=False,
                 perf_mode="exp", eps=1e-6, perf_alpha=1.0):
    # 1. 复制参数
    parameters = copy.deepcopy(dict(model.named_parameters()))

    # 2. 复制输入并开启梯度
    # 注意：这里的 feat_t/a/v 已经是 Backbone 提取出的 128 维向量
    feat_t = copy.deepcopy(feat_t.detach()).float().requires_grad_(True)
    feat_a = copy.deepcopy(feat_a.detach()).float().requires_grad_(True)
    feat_v = copy.deepcopy(feat_v.detach()).float().requires_grad_(True)

    # 3. 【核心修改】直接从层对象获取权重，兼容 Spectral Norm
    # 以前是查字典 parameters['...weight']，现在改成直接访问对象属性
    if hasattr(model, 'module'):
        fusion_layer = model.module.fusion_model.fxy
    else:
        fusion_layer = model.fusion_model.fxy

    # 这里的 .weight 属性在 Spectral Norm 下会自动返回计算后的有效权重
    # 不需要去管它是叫 weight 还是 weight_orig
    W = fusion_layer.weight.float()

    # 获取偏置
    if fusion_layer.bias is not None:
        b_full = fusion_layer.bias.float()
        b_partial = (b_full / 3.0).detach().requires_grad_(True)
        B_total = b_full.detach().requires_grad_(True)
    else:
        # 防御性编程：万一没偏置
        b_partial = 0
        B_total = 0

    # 4. 切分权重
    # 动态读取 map_dim，避免写死 128
    if hasattr(model, 'module') and hasattr(model.module, 'map_dim'):
        dim = int(model.module.map_dim)
    elif hasattr(model, 'map_dim'):
        dim = int(model.map_dim)
    else:
        dim = 128

    # 注意：W 的形状是 [1, 384] 或者 [Out, In]
    # Linear 层的 weight 形状通常是 [out_features, in_features]
    # 我们的输入是 concat(t, a, v)，所以权重在第 1 维切分
    W_t = W[:, :dim].detach().requires_grad_(True)
    W_a = W[:, dim:2 * dim].detach().requires_grad_(True)
    W_v = W[:, 2 * dim:].detach().requires_grad_(True)

    w_total = W.detach().requires_grad_(True)
    # # 3. 提取融合层权重
    # # 路径根据 models/multimodal.py 里的定义：fusion_model.fxy
    # if hasattr(model, 'module'):
    #     weight_key = 'module.fusion_model.fxy.weight'
    #     bias_key = 'module.fusion_model.fxy.bias'
    # else:
    #     weight_key = 'fusion_model.fxy.weight'
    #     bias_key = 'fusion_model.fxy.bias'
    #
    # W = parameters[weight_key].double()
    # # 偏置除以3 (因为有3个模态贡献)
    # b_partial = (parameters[bias_key].double() / 3.0).detach().requires_grad_(True)
    # B_total = (parameters[bias_key].double()).detach().requires_grad_(True)
    #
    # # 4. 切分权重
    # # 我们约定的 map_dim 是 128
    # dim = 128
    # W_t = W[:, :dim].detach().requires_grad_(True)
    # W_a = W[:, dim:2 * dim].detach().requires_grad_(True)
    # W_v = W[:, 2 * dim:].detach().requires_grad_(True)
    # w_total = W.detach().requires_grad_(True)

    # 5. 计算单模态预测 (Logits / Score)
    # F.linear(input, weight, bias) -> y = xW^T + b
    pred_t = F.linear(feat_t, W_t, b_partial)
    pred_a = F.linear(feat_a, W_a, b_partial)
    pred_v = F.linear(feat_v, W_v, b_partial)
    pred_f = F.linear(torch.cat([feat_t, feat_a, feat_v], dim=1), w_total, B_total)

    # 6. 计算验证集准确率 (可选，针对二分类)
    if Val == True:
        # 假设做二分类准确率统计
        # 如果是回归任务，这里可以改返回 MAE
        return 0, 0, 0

        # 7. 计算 Loss
    # 注意：MOSI 是回归，Label 是 Float，Criterion 应该是 L1Loss
    # 为了保证梯度计算正常，label 需要 reshape 成 [B, 1]
    if len(label.shape) == 1:
        label = label.view(-1, 1)

    l_t = criterion(pred_t, label)
    l_a = criterion(pred_a, label)
    l_v = criterion(pred_v, label)
    l_f = criterion(pred_f, label)

    # 8. 计算梯度 (保留用于分析，如余弦相似度)
    # 这里我们主要为了 backward，grad_list 可以简化
    grad_t = list(torch.autograd.grad(l_t, [feat_t, W_t, b_partial], create_graph=True, allow_unused=True))
    grad_a = list(torch.autograd.grad(l_a, [feat_a, W_a, b_partial], create_graph=True, allow_unused=True))
    grad_v = list(torch.autograd.grad(l_v, [feat_v, W_v, b_partial], create_graph=True, allow_unused=True))

    # 9. 反向传播
    l_t.backward()
    l_a.backward()
    l_v.backward()
    l_f.backward()

    # # 10. 计算 Performance (贡献度)
    # # 对于回归任务，Loss 越小越好。为了兼容 GMML 的逻辑(数值越大越强)，我们取倒数
    # # 加个 epsilon 防止除以 0
    # perf_t = 1.0 / (l_t.item() + 1e-6)
    # perf_a = 1.0 / (l_a.item() + 1e-6)
    # perf_v = 1.0 / (l_v.item() + 1e-6)
    # 10. 计算 Performance (贡献度)  —— 支持 exp / inv
    mode = (perf_mode or "exp").lower()

    # 这里用 detach，避免 perf 分支产生旁路梯度
    lt = float(l_t.detach().item())
    la = float(l_a.detach().item())
    lv = float(l_v.detach().item())

    if mode == "exp":
        # p = exp(-alpha * L)
        perf_t = math.exp(-perf_alpha * lt)
        perf_a = math.exp(-perf_alpha * la)
        perf_v = math.exp(-perf_alpha * lv)
    elif mode == "inv":
        # p = 1/(L+eps)
        perf_t = 1.0 / (lt + eps)
        perf_a = 1.0 / (la + eps)
        perf_v = 1.0 / (lv + eps)
    else:
        raise ValueError(f"Unknown perf_mode: {perf_mode}, expected 'exp' or 'inv'")

    # 组织返回值
    # param_dict 用于 train.py 里的参数约束
    param_dict = {'text': W_t, 'audio': W_a, 'visual': W_v}

    # 返回: sim(暂时没算), grad_list, perfs..., losses...
    return 0, [grad_t, grad_a, grad_v], perf_t, perf_a, perf_v, l_t, l_a, l_v


def calculate_cosine_similarity(grad1, grad2):
    if isinstance(grad1, tuple): grad1 = torch.tensor(grad1)
    if isinstance(grad2, tuple): grad2 = torch.tensor(grad2)
    grad1_flat = grad1.reshape(-1)
    grad2_flat = grad2.reshape(-1)
    if grad1_flat.size(0) != grad2_flat.size(0): return 0
    cos_sim = F.cosine_similarity(grad1_flat, grad2_flat, dim=0)
    return cos_sim.item()


def EMA(befor, now, t, alpha):
    return ((1 - alpha) * now + (alpha) * befor)

# 其他辅助函数保持不变...

# ---------------------------------------------------------------------
# 请将以下代码追加到 util/util.py 文件中，补全缺失的工具类
# ---------------------------------------------------------------------

class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(save_file_path, epoch, model, optimizer, scheduler):
    """保存模型检查点"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(save_states, save_file_path)
