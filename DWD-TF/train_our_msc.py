import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加这行
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset_msc import (BaseDataSets, RandomGenerator,TwoStreamBatchSampler)
from models.net_factory import net_factory
from utils import losses, metrics, ramps, val_2d
from utils.Generate_Prototype import *

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_enhanced_consistency_weight(epoch, mixing_ratio):
    """调整一致性权重以适应UA-DTCT"""
    base_weight = args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

    # 根据UA-DTCT强度调整
    if mixing_ratio > 0.5:
        return base_weight * 0.9  # 轻微降低
    else:
        return base_weight

def generate_threshold(con, proportion):

    k = int(con.numel() * proportion)
    lowestk_con, _ = torch.topk(con.view(-1), k, largest=False)
    proportion_threshold = torch.mean(lowestk_con)
    return proportion_threshold

def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

def get_ua_dtct_mixing_ratio(iter_num, total_iterations=30000):
    """更保守的混合策略"""
    if iter_num < 5000:
        return 0.0  # 前5000次完全使用原方法
    elif iter_num < 10000:
        return 0.3  # 逐步引入30%
    elif iter_num < 15000:
        return 0.6  # 增加到60%
    elif iter_num < 20000:
        return 0.8  # 增加到80%
    else:
        return 1.0  # 最后阶段完全使用


def get_adaptive_temperature(iter_num, base_temp=0.1):
    """更精细的温度调整策略"""
    if iter_num < 3000:
        return 1.0  # 早期：很高温度，权重接近平均
    elif iter_num < 8000:
        return 0.5  # 中期前：中等温度
    elif iter_num < 15000:
        return 0.3  # 中期：较低温度
    elif iter_num < 20000:
        return 0.2  # 中后期
    else:
        return 0.15  # 后期：稍高于原温度


def uncertainty_aware_pseudo_label(outputs_soft1, outputs_soft2, max_value1, max_value2, temperature=0.1):
    """更稳定的不确定性感知伪标签融合"""

    # 计算两个模型的不确定性（熵）
    uncertainty1 = -torch.sum(outputs_soft1 * torch.log(outputs_soft1 + 1e-6), dim=1, keepdim=True)
    uncertainty2 = -torch.sum(outputs_soft2 * torch.log(outputs_soft2 + 1e-6), dim=1, keepdim=True)

    # 限制不确定性的范围
    uncertainty1 = torch.clamp(uncertainty1, min=1e-6, max=2.0)
    uncertainty2 = torch.clamp(uncertainty2, min=1e-6, max=2.0)

    # 使用置信度调整不确定性
    conf1 = max_value1.unsqueeze(1)
    conf2 = max_value2.unsqueeze(1)

    # 更温和的置信度调整
    adjusted_uncertainty1 = uncertainty1 * (1.0 - conf1 * 0.5)
    adjusted_uncertainty2 = uncertainty2 * (1.0 - conf2 * 0.5)

    # 使用更稳定的权重计算
    weights = torch.stack([
        torch.exp(-adjusted_uncertainty1 / (temperature + 0.1)),
        torch.exp(-adjusted_uncertainty2 / (temperature + 0.1))
    ], dim=1)

    # 更宽松的最小权重
    min_weight = args.min_weight
    max_weight = args.max_weight
    weights = torch.clamp(weights, min=min_weight, max=max_weight)

    # 重新归一化
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    weight1 = weights[:, 0:1]
    weight2 = weights[:, 1:2]

    # 加权融合伪标签
    fused_pseudo = weight1 * outputs_soft1 + weight2 * outputs_soft2

    return fused_pseudo, weight1, weight2


def get_adaptive_loss_weights(iter_num):
    """根据训练阶段调整各损失项的权重"""
    if iter_num < 5000:
        focus_scale = 0.8
        pro_scale = 0.8
    elif iter_num < 15000:
        focus_scale = 1.0
        pro_scale = 1.0
    else:
        focus_scale = 1.1
        pro_scale = 1.1

    return focus_scale, pro_scale

def compute_prototypes(fea1, fea2, max_index1, max_index2, num_classes,flag=1):

    if flag == 0:
        fts1 = F.interpolate(fea1, size=max_index1.shape[-2:], mode='bilinear')
        fts2 = F.interpolate(fea2, size=max_index2.shape[-2:], mode='bilinear')
        return  fts1, fts2

    if flag==1:

        one_hot1 = torch.nn.functional.one_hot(max_index1, num_classes=num_classes)
        one_hot2 = torch.nn.functional.one_hot(max_index2, num_classes=num_classes)

        one_hot1 = one_hot1.permute(0, 3, 1, 2)
        one_hot2 = one_hot2.permute(0, 3, 1, 2)

        fts1 = F.interpolate(fea1, size=max_index1.shape[-2:], mode='bilinear')
        fts2 = F.interpolate(fea2, size=max_index2.shape[-2:], mode='bilinear')

        prototypes1 = getPrototype_2D(fts1, one_hot1)
        prototypes2 = getPrototype_2D(fts2, one_hot2)

        return prototypes1, prototypes2 ,fts1, fts2

def compute_prototype_loss(fts1, fts2, prototypes1, prototypes2, max_index1, max_index2, high_low_con_mask1,
                           high_low_con_mask2, scaler):

    pro_cos1 = torch.stack(
        [calDist_2D(fts1, prototype2, scaler=scaler) for prototype2 in prototypes2], dim=1)
    pro_cos2 = torch.stack(
        [calDist_2D(fts2, prototype1, scaler=scaler) for prototype1 in prototypes1], dim=1)

    loss_pro_ce1 = F.cross_entropy(pro_cos1, max_index2, reduction='none')
    loss_pro_ce2 = F.cross_entropy(pro_cos2, max_index1, reduction='none')

    loss_pro_ce1 = torch.sum(high_low_con_mask1.unsqueeze(1) * loss_pro_ce1) / (
            torch.sum(high_low_con_mask1.unsqueeze(1)) + 1e-16)
    loss_pro_ce2 = torch.sum(high_low_con_mask2.unsqueeze(1) * loss_pro_ce2) / (
            torch.sum(high_low_con_mask2.unsqueeze(1)) + 1e-16)

    loss_pro_cos1 = ((1 - pro_cos1) * high_low_con_mask1.unsqueeze(1)).mean()
    loss_pro_cos2 = ((1 - pro_cos2) * high_low_con_mask2.unsqueeze(1)).mean()

    return loss_pro_ce1, loss_pro_ce2, loss_pro_cos1, loss_pro_cos2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='D:\\HaotianZhang\\ALHVR.-main\\ALHVR.-main\\train_ALHVR\\data\\msc', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DWD-TF', help='experiment_name')
parser.add_argument('--model', type=str, default='unet_fea_aux', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--proportion', type=float, default=0.8, help='***')
parser.add_argument('--scaler', type=float, default=1, help='multiplier of prototype')
# 新增调优参数
parser.add_argument('--ua_warmup', type=int, default=10000, help='UA-DTCT warmup iterations')
parser.add_argument('--min_weight', type=float, default=0.35, help='minimum weight for UA-DTCT')
parser.add_argument('--max_weight', type=float, default=0.65, help='maximum weight for UA-DTCT')
parser.add_argument('--conf_adjust', type=float, default=0.5, help='confidence adjustment strength')
args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if 'msc' in dataset:
        ref_dict = {'7': 110}  # split1
        # elif 'split2' in dataset:
        #     ref_dict = {'7': 103}
    else:
        raise NotImplementedError
    return ref_dict[str(patiens_num)]

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model1 = net_factory(net_type="unet_fea_aux", in_chns=1, class_num=num_classes)
    model2 = net_factory(net_type="unet_fea_aux", in_chns=1, class_num=num_classes)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model1.train()
    model2.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model1.train()
            model2.train()

            outputs1,outputs_aux1, fea1, fea_aux1 = model1(volume_batch)
            outputs2,outputs_aux2, fea2, fea_aux2 = model2(volume_batch)

            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)
            outputs_soft_aux1 = F.softmax(outputs_aux1, dim=1)
            outputs_soft_aux2 = F.softmax(outputs_aux2, dim=1)

            loss_dice1 = dice_loss(outputs_soft1[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_dice2 = dice_loss(outputs_soft2[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_ce1 = F.cross_entropy(outputs1[:labeled_bs], label_batch[:labeled_bs].long())
            loss_ce2 = F.cross_entropy(outputs2[:labeled_bs], label_batch[:labeled_bs].long())

            # 监督损失
            loss_sup1 = loss_dice1  + loss_ce1
            loss_sup2 = loss_dice2  + loss_ce2

            #consistency_weight = get_current_consistency_weight(iter_num // 150)
            mixing_ratio = get_ua_dtct_mixing_ratio(iter_num, args.max_iterations)
            consistency_weight = get_enhanced_consistency_weight(iter_num // 150, mixing_ratio)
            max_value_aux1, max_index_aux1 = outputs_soft_aux1[labeled_bs:].max(dim=1)
            max_value_aux2, max_index_aux2 = outputs_soft_aux2[labeled_bs:].max(dim=1)
            max_value1, max_index1 = outputs_soft1[labeled_bs:].max(dim=1)
            max_value2, max_index2 = outputs_soft2[labeled_bs:].max(dim=1)

            threshold = generate_threshold((max_value_aux1 + max_value_aux2) * 0.5, args.proportion)
            low_con_mask = ((max_value_aux1 < threshold) & (max_value_aux2 < threshold)).to(torch.int32)

            high_low_con_mask1 = ((max_value_aux1 <= threshold) & (max_value_aux2 > threshold)).to(torch.int32)
            high_low_con_mask2 = ((max_value_aux2 <= threshold) & (max_value_aux1 > threshold)).to(torch.int32)

            # ========== 改进的UA-DTCT实现 ==========
            # 获取混合比例和自适应温度
            adaptive_temperature = get_adaptive_temperature(iter_num, args.temperature)

            original_pseudo = torch.where((max_value1 > max_value2).unsqueeze(1), outputs_soft1[labeled_bs:],outputs_soft2[labeled_bs:])

            # 新的不确定性感知融合方法
            fused_pseudo, weight1, weight2 = uncertainty_aware_pseudo_label(
                outputs_soft1[labeled_bs:],
                outputs_soft2[labeled_bs:],
                max_value1,
                max_value2,
                temperature=adaptive_temperature  # 复用现有的温度参数
            )
            # 混合伪标签
            if mixing_ratio < 1.0:
                final_pseudo = mixing_ratio * fused_pseudo + (1 - mixing_ratio) * original_pseudo
            else:
                final_pseudo = fused_pseudo
            # 锐化融合伪标签
            sharpened_fused_pseudo = sharpening(final_pseudo)

            # 记录调试信息
            if iter_num % 200 == 0:
                avg_weight1 = torch.mean(weight1).item()
                avg_weight2 = torch.mean(weight2).item()
                mixing_info = f"Mixing: {mixing_ratio:.3f}, Temp: {adaptive_temperature:.3f}"
                weight_info = f"Weights - Model1: {avg_weight1:.3f}, Model2: {avg_weight2:.3f}"
                logging.info(f"Iter {iter_num}: {mixing_info}, {weight_info}")
            # ========== UA-DTCT结束 ==========


            mse_dist1 = (outputs_soft_aux1[labeled_bs:] - sharpened_fused_pseudo) ** 2
            mse1 = torch.sum(low_con_mask.unsqueeze(1) * mse_dist1) / (torch.sum(low_con_mask.unsqueeze(1)) + 1e-16)
            mse_dist2 = (outputs_soft_aux2[labeled_bs:] - sharpened_fused_pseudo) ** 2
            mse2 = torch.sum(low_con_mask.unsqueeze(1) * mse_dist2) / (torch.sum(low_con_mask.unsqueeze(1)) + 1e-16)

            preds1 = outputs_soft1[labeled_bs:] * low_con_mask.unsqueeze(1)
            preds_aux1 = outputs_soft_aux1[labeled_bs:] * low_con_mask.unsqueeze(1)
            preds2 = outputs_soft2[labeled_bs:] * low_con_mask.unsqueeze(1)
            preds_aux2 = outputs_soft_aux2[labeled_bs:] * low_con_mask.unsqueeze(1)

            uncertainty1 = -1.0 * \
                          torch.sum(preds1 * torch.log(preds1 + 1e-6), dim=1, keepdim=True)
            uncertainty2 = -1.0 * \
                           torch.sum(preds2 * torch.log(preds2 + 1e-6), dim=1, keepdim=True)
            uncertainty_aux1 = -1.0 * \
                           torch.sum(preds_aux1 * torch.log(preds_aux1 + 1e-6), dim=1, keepdim=True)
            uncertainty_aux2 =  -1.0 * \
                           torch.sum(preds_aux2 * torch.log(preds_aux2 + 1e-6), dim=1, keepdim=True)
            loss_focus1 = mse1 + torch.mean(uncertainty1) + torch.mean(uncertainty_aux1)
            loss_focus2 = mse2 + torch.mean(uncertainty2) + torch.mean(uncertainty_aux2)

            prototypes1,prototypes2,fts1,fts2 = compute_prototypes(fea1[labeled_bs:], fea2[labeled_bs:], max_index1, max_index2, num_classes,1)
            fts_aux1, fts_aux2 = compute_prototypes(fea_aux1[labeled_bs:], fea_aux2[labeled_bs:], max_index_aux1, max_index_aux2, num_classes,0)
            loss_pro_ce1, loss_pro_ce2, loss_pro_cos1, loss_pro_cos2 = compute_prototype_loss(fts_aux1, fts_aux2, prototypes1,
                                                                                              prototypes2, max_index1,
                                                                                              max_index2,
                                                                                              high_low_con_mask1,
                                                                                              high_low_con_mask2,
                                                                                              args.scaler)

            loss_pro1 = loss_pro_ce1+loss_pro_cos1
            loss_pro2 = loss_pro_ce2+loss_pro_cos2

            # 在损失计算中添加权重调整：
            focus_scale, pro_scale = get_adaptive_loss_weights(iter_num)
            loss1 = loss_sup1 + consistency_weight * (focus_scale * loss_focus1 + pro_scale * loss_pro1 )
            loss2 = loss_sup2 + consistency_weight * (focus_scale * loss_focus2 + pro_scale * loss_pro2 )
            loss = loss1 + loss2

            iter_num = iter_num + 1
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            # ========== 添加可视化日志 ==========
            if iter_num % 50 == 0:  # 每50次迭代记录一次权重信息
                avg_weight1 = torch.mean(weight1).item()
                avg_weight2 = torch.mean(weight2).item()
                weight_diff = abs(avg_weight1 - avg_weight2)

                logging.info(
                    f'Iter {iter_num}: UA-DTCT weights - Model1: {avg_weight1:.4f}, Model2: {avg_weight2:.4f}, Diff: {weight_diff:.4f}')

            logging.info(
                'iteration %d : loss: %03f ,loss_sup1: %03f,loss_sup2: %03f,loss_pro1: %03f,loss_pro2: %03f,loss_focus1: %03f,loss_focus2: %03f' % (
                    iter_num, loss, loss_sup1,loss_sup2, loss_pro1,loss_pro2,loss_focus1,loss_focus2))

            if iter_num > 0 and iter_num % 200== 0:
                # ========== 详细的可视化日志 ==========
                # UA-DTCT权重统计
                avg_weight1 = torch.mean(weight1).item()
                avg_weight2 = torch.mean(weight2).item()
                weight_diff = abs(avg_weight1 - avg_weight2)

                writer.add_scalar('UA_DTCT/weight_model1', avg_weight1, iter_num)
                writer.add_scalar('UA_DTCT/weight_model2', avg_weight2, iter_num)
                writer.add_scalar('UA_DTCT/weight_difference', weight_diff, iter_num)

                # 添加混合策略信息
                writer.add_scalar('UA_DTCT/mixing_ratio', mixing_ratio, iter_num)
                writer.add_scalar('UA_DTCT/adaptive_temperature', adaptive_temperature, iter_num)

                # 不确定性统计
                avg_uncertainty1 = torch.mean(uncertainty1).item()
                avg_uncertainty2 = torch.mean(uncertainty2).item()
                writer.add_scalar('UA_DTCT/uncertainty_model1', avg_uncertainty1, iter_num)
                writer.add_scalar('UA_DTCT/uncertainty_model2', avg_uncertainty2, iter_num)

                # 区域分布统计
                low_con_ratio = torch.mean(low_con_mask.float()).item()
                high_low_ratio1 = torch.mean(high_low_con_mask1.float()).item()
                high_low_ratio2 = torch.mean(high_low_con_mask2.float()).item()

                writer.add_scalar('Regions/low_confidence_ratio', low_con_ratio, iter_num)
                writer.add_scalar('Regions/high_low_region1_ratio', high_low_ratio1, iter_num)
                writer.add_scalar('Regions/high_low_region2_ratio', high_low_ratio2, iter_num)
                writer.add_scalar('Regions/total_unlabeled_ratio',
                                  low_con_ratio + high_low_ratio1 + high_low_ratio2, iter_num)

                # 动态阈值记录
                writer.add_scalar('Threshold/dynamic_threshold', threshold.item(), iter_num)

                # 一致性权重记录
                writer.add_scalar('Training/consistency_weight', consistency_weight, iter_num)

                logging.info(
                    f'Region Stats - LowConf: {low_con_ratio:.3f}, HighLow1: {high_low_ratio1:.3f}, HighLow2: {high_low_ratio2:.3f}')
                logging.info(f'Uncertainty - Model1: {avg_uncertainty1:.3f}, Model2: {avg_uncertainty2:.3f}')

                model1.eval()
                metric_list1 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i1 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model1,
                                                         classes=num_classes)
                    metric_list1 += np.array(metric_i1)
                metric_list1 = metric_list1 / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list1[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list1[class_i, 1], iter_num)

                performance1 = np.mean(metric_list1, axis=0)[0]
                mean_hd951 = np.mean(metric_list1, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path1 = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_model1.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best_path1 = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path1)
                    torch.save(model1.state_dict(), save_best_path1)
                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list2 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i2 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model2,
                                                          classes=num_classes)
                    metric_list2 += np.array(metric_i2)
                metric_list2 = metric_list2 / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list2[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list2[class_i, 1], iter_num)

                performance2 = np.mean(metric_list2, axis=0)[0]
                mean_hd952 = np.mean(metric_list2, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path2 = os.path.join(snapshot_path,
                                                   'iter_{}_dice_{}_model2.pth'.format(iter_num,
                                                                                       round(best_performance2, 4)))
                    save_best_path2 = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path2)
                    torch.save(model2.state_dict(), save_best_path2)
                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/mscmrseg19_{}_labeled/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('./code/', snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
