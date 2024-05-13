import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from collections import OrderedDict
import copy
import json
import random
from torch.optim.lr_scheduler import MultiStepLR 
import torch.optim as optim
from tqdm import tqdm


#####################################################################
#                    sparse mask generation                         #
#####################################################################
def generate_sparse_mask(model, sparse_ratio):
    return weight_magnitude_based_sc(model, sparse_ratio, 'lwf')
    
def weight_magnitude_based_sc(model, sr, sc):
    '''
    weight_magnitude_based_selection_criteria.
    '''
    compare = torch.gt if sc == 'lwf' else torch.lt  # torch.gt（大于）或 torch.lt（小于）
    sr = sr if sc == 'lwf' else (1-sr)  # 选择率
    
    weights_values = torch.Tensor([])
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):  # 判断当前的模块 m 是否是线性层（nn.Linear）或卷积层（nn.Conv2d）
            # 使用 torch.concat 函数将新展平并克隆的权重张量与已经收集的 weights_values 张量合并。weights_values 在每次迭代中都会更新，以包含到目前为止检查过的所有层的权重。
            weights_values = torch.concat((weights_values, m.weight.data.abs().clone().view(-1)))  # .view(-1): 这会改变张量的形状，将其展平为一维张量

    n = int(len(weights_values)*sr)
    sorted_values, _ = torch.sort(weights_values, descending=True)  # 降序
    # sorted_values, _ = torch.sort(weights_values)
    threshold = sorted_values[n-1]  # 将第n个值作为阈值
    sparse_masks = []
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = compare(weight_copy, threshold).float()  # mask不是1.就是0.
            sparse_masks.append(mask)  
    return sparse_masks



def init_weights(model, random_seed=None):
    if random_seed != None:
        torch.manual_seed(random_seed)            
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # m.weight.data *= 0.3
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# for m in self.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)


def norm_(grads_list):
    max_num = torch.max(grads_list[0])
    for idx in range(1, len(grads_list)):
        max_in_cur_layer = torch.max(grads_list[idx])
        max_num = max_num if max_in_cur_layer <= max_num else max_in_cur_layer
    
    for idx in range(len(grads_list)):
        # norm to [0, 1]
        grads_list[idx] = grads_list[idx]/max_num
    return grads_list


#####################################################################
#                         pluggable adapter                         #
#####################################################################
def reverse_mask(masks):
    # mask: a list contain several mask of conv or bn layer
    tmp = copy.deepcopy(masks)  # 深拷贝确保了对 tmp 的任何修改都不会影响原始的 masks 列表。
    for idx in range(len(masks)):
        tmp[idx] = 1. - masks[idx]  # 对于 masks 中的每一个掩码，函数使用表达式 1. - masks[idx] 来计算其反转版本。
    return tmp


def insert_adapter(model, sparse_mask, model_seed, is_sparse=True):
    """
    model_seed: the model whose weight are initialized according to key(random seed).
    """
    reverse_sparse_mask = reverse_mask(sparse_mask)
    idx_m = 0
    for [m, m_s] in zip(model.modules(), model_seed.modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if is_sparse == False:
                m.weight.data = m.weight.data.mul_(sparse_mask[idx_m])
            m.weight.data += m_s.weight.data.clone().mul_(reverse_sparse_mask[idx_m])
            idx_m += 1


def remove_adapter(model, sparse_mask):
    idx_m = 0
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.mul_(sparse_mask[idx_m])  # 这里的 sparse_mask 通常是一个由0和1组成的张量，用于指示哪些权重应该被保留（值为1）和哪些应该被设置为零（值为0），从而实现模型权重的稀疏化
            idx_m += 1


def insert_adapter_for_receiver(model, model_seed):
    idx_m = 0
    for [(k, m), (k_s, m_s)] in zip(model.named_modules(), model_seed.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            cur_reverse_mask  = m.weight.data.eq(0.).float()
            m.weight.data += m_s.weight.data.clone().mul_(cur_reverse_mask)
            idx_m += 1


def remove_adapter_for_receiver(model, model_seed):
    idx_m = 0
    for [(k, m), (k_s, m_s)] in zip(model.named_modules(), model_seed.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            equal_pos = (m.weight.data == m_s.weight.data).float()
            m.weight.data = m.weight.data.mul_(1. - equal_pos)
            idx_m += 1


# #####################################################################
# #                         batch normalization                       #
# #####################################################################

# def save_bn(model):
#     r_ms = []; r_vs = []
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             r_ms.append(m.running_mean.clone())
#             r_vs.append(m.running_var.clone())
#     return r_ms, r_vs


# def restore_bn(model, r_ms, r_vs):
#     idx = 0
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.running_mean = r_ms[idx]
#             m.running_var = r_vs[idx]
#             idx += 1


# def record_bn_into_json(b_rm, b_rv, file_path):

#     for idx in range(len(b_rm)):
#         b_rm[idx] = b_rm[idx].tolist()
#         b_rv[idx] = b_rv[idx].tolist()

#     fs_se = {"b_rm_se": b_rm, "b_rv_se": b_rv}
#     jsonString = json.dumps(fs_se)
#     jsonFile = open(file_path, "w")
#     jsonFile.write(jsonString)
#     jsonFile.close()
 

# def load_bn_from_json(file_path, device): 
#     fileObject = open(file_path, "r")
#     jsonContent = fileObject.read()
#     fs_se = json.loads(jsonContent)
#     b_rm_se = fs_se['b_rm_se']; b_rv_se = fs_se['b_rv_se']

#     for idx in range(len(b_rm_se)):
#         b_rm_se[idx] = torch.from_numpy(np.array(b_rm_se[idx])).float().to(device)
#         b_rv_se[idx] = torch.from_numpy(np.array(b_rv_se[idx])).float().to(device)
    
#     return b_rm_se, b_rv_se

