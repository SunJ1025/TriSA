# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import time
import os
from model import two_view_net
import yaml
from shutil import copyfile
from tools.utils import save_network, setup_seed
from losses.triplet_loss import Tripletloss
from losses.cal_loss import cal_triplet_loss
from losses.cross_entroy_loss import cross_entropy_loss
from pytorch_metric_learning import losses, miners
from triplet_samp_load.Get_DataLoader import get_data_loader

version = torch.__version__
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--name', default='test_final', type=str, help='output model name')
parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--num_epochs', default=120, type=int, help='')

parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')

parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--sample_num', default=4, type=float, help='num of repeat sampling')
parser.add_argument('--margin', default=0.3, type=float, help='num of margin')

parser.add_argument('--labelsmooth', default=0, type=int, help='1表示使用 0表示不使用')
parser.add_argument('--share', action='store_true', help='share weight between different view')

parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')  # 设置 FC层的 drop rate
parser.add_argument('--stride', default=1, type=int, help='stride')  # 网络最后层下采样的步长 默认为2

parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=[70, 110], type=int, help='')  # 学习率突变的 epochs
parser.add_argument('--balance', default=1.0, type=float, help='balance rate for triplet loss')  # 平衡三元组损失正则化项

# action='store_true' 如果添加 --triplet 就表示使用
parser.add_argument('--triplet', action='store_true', help='use triplet loss')
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
parser.add_argument('--cosface', action='store_true', help='use CosFace loss')
parser.add_argument('--contrast', action='store_true', help='use contrast loss')
parser.add_argument('--lifted', action='store_true', help='use lifted loss')
parser.add_argument('--sphere', action='store_true', help='use sphere loss')
opt = parser.parse_args()

# 设置 GPU
torch.cuda.set_device(0)
cudnn.benchmark = True
setup_seed()

dataloaders, opt.nclasses, dataset_sizes = get_data_loader(opt.h, opt.w, opt.pad, opt.erasing_p, opt.color_jitter,
                                                           opt.DA, opt.data_dir, opt.sample_num, opt.batchsize)
print("类别数：", opt.nclasses)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):  # 训练模型

    begin = time.time()

    warm_up = 0.1  # warm_up 训练策略设置
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
    MAX_LOSS = 10

    if opt.triplet:
        triplet_loss = Tripletloss(margin=opt.margin, balance=opt.balance)  # 域间三元组
        miner = miners.MultiSimilarityMiner()  # 域内三元组
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        since = time.time()
        model.train(True)

        running_cls_loss = 0.0  # 每一个 epoch 将 loss 清零
        running_triplet = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0

        # 开始小的Iterate
        for data, data2 in dataloaders:

            inputs, labels, _ = data  # 卫星数据
            inputs, labels = Variable(inputs.cuda().detach()), Variable(labels.cuda().detach())

            inputs2, labels2, label_id_dr = data2  # 无人机数据
            inputs2, labels2 = Variable(inputs2.cuda().detach()), Variable(labels2.cuda().detach())

            now_batch_size, _, _, _ = inputs.shape
            if now_batch_size < opt.batchsize:
                continue
            optimizer.zero_grad()  # 梯度清零

            outputs, outputs2 = model(inputs, inputs2)  # 返回概率值和特征
            logits, ff = outputs
            logits2, ff2 = outputs2

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)  # 特征归一化
            fnorm2 = torch.norm(ff2, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff2 = ff2.div(fnorm2.expand_as(ff2))

            _, preds = torch.max(logits.data, 1)
            _, preds2 = torch.max(logits2.data, 1)
            cls_loss = criterion(logits, labels) + criterion(logits2, labels2)

            if isinstance(preds, list) and isinstance(preds2, list):
                print("yes")
            if opt.triplet:  # 是否使用三元组损失
                f_triplet_loss = cal_triplet_loss(ff, ff2, labels, labels2, triplet_loss)

                # 加上域内的三元组损失  
                hard_pairs = miner(ff, labels)
                # hard_pairs2 = miner(ff2, labels2)           
                f_triplet_loss = criterion_triplet(ff, labels,
                                                   hard_pairs) + f_triplet_loss  # + criterion_triplet(ff2, labels2, hard_pairs2)

                loss = cls_loss * 0.5 + f_triplet_loss
            else:
                loss = cls_loss

            if epoch < opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up

            loss.backward()
            optimizer.step()

            # 记录数据
            if int(version[0]) > 0 or int(version[2]) > 3:
                running_loss += loss.item() * now_batch_size
            else:
                running_loss += loss.data[0] * now_batch_size

            running_cls_loss += cls_loss * now_batch_size
            running_triplet += f_triplet_loss * now_batch_size

            running_corrects += float(torch.sum(preds == labels.data))
            running_corrects2 += float(torch.sum(preds2 == labels2.data))

        scheduler.step()
        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_cls_loss = running_cls_loss / dataset_sizes['satellite']
        epoch_triplet_loss = running_triplet / dataset_sizes['satellite']

        epoch_acc = running_corrects / dataset_sizes['satellite']  # epoch 卫星正确率
        epoch_acc2 = running_corrects2 / dataset_sizes['satellite']  # epoch 无人机正确率
        print(
            'Epoch: {}  Loss: {:.4f} Cls_Loss:{:.4f}  Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_cls_loss, epoch_triplet_loss, epoch_acc, epoch_acc2))

        if epoch_acc > 0.8 and epoch_acc2 > 0.8:
            if epoch_loss < MAX_LOSS and epoch > (num_epochs - 40) or epoch == num_epochs - 1:
                MAX_LOSS = epoch_loss
                save_network(model, opt.name, epoch)
                print(opt.name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))

        time_elapsed = time.time() - since
        print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    time_elapsed = time.time() - begin
    print('Total training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere
    model = two_view_net(class_num=opt.nclasses, droprate=opt.droprate, stride=opt.stride, share_weight=opt.share,
                         circle=return_feature)

    ignored_params = list(map(id, model.classifier.parameters()))  # 全连接层和其他的层是不一样的学习率
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    criterion = nn.CrossEntropyLoss()
    if opt.labelsmooth == 1:
        print("use label smooth")
        criterion = cross_entropy_loss()

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.steps, gamma=0.1)

    dir_name = os.path.join('checkpoints', opt.name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    copyfile('train.py', dir_name + '/train.py')
    copyfile('./model.py', dir_name + '/model.py')

    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    model = train_model(model.cuda(), criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.num_epochs)  # 调用训练函数
