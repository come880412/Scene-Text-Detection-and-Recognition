'''
Modified Date: 2021/12/20

Author: Gi-Luen Huang

mail: come880412@gmail.com
'''

import os
import sys
import time
import random
import string
import argparse
import re
import torch.nn as nn
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import tqdm

from utils import Averager, TokenLabelConverter
from dataset import RawDataset
from model import Model
from utils import get_args

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    """ dataset preparation """
    train_dataset = RawDataset(opt, mode='train')
    val_dataset = RawDataset(opt, mode='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, 
        num_workers=int(opt.workers))
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size,
        shuffle=True, 
        num_workers=int(opt.workers))

    opt.eval = True

    """ Prepare the charcter list"""
    opt.character = ''
    with open(opt.character_dir, encoding='utf-8') as f:
        data = f.read().splitlines()
        for line in data:
            opt.character = opt.character + line
    """ model configuration """
    converter = TokenLabelConverter(opt)

    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    """ Load pretrained model"""
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model))
    model = model.to(device)

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    # print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.sgd:
        optimizer = optim.SGD(filtered_parameters, lr=opt.lr, momentum=0.9)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a', encoding='utf-8') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    for epoch in range(opt.n_epochs):
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        for data in train_loader:
            # train part
            image_tensors, labels = data[0], data[1]
            image = image_tensors.to(device)
            target = converter.encode(labels)

            preds, _ = model(image, seqlen=converter.batch_max_length)
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_avg.val():.4f}",
            )
        pbar.close()

        # validation part
        elapsed_time = time.time() - start_time
        # for log
        with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
            model.eval()
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                    model, criterion, valid_loader, converter, opt)
            model.train()

            # training loss and validation loss
            loss_log = f'[{epoch}/{opt.n_epochs}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()

            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

            # keep best accuracy model (on valid dataset)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
                torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)
            log.write(loss_model_log + '\n')

            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                pred = pred[:pred.find('[s]')]
                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
            log.write(predicted_result_log + '\n')

        if scheduler is not None:
            scheduler.step()

def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        target = converter.encode(labels)

        start_time = time.time()
        preds, _ = model(image, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        forward_time = time.time() - start_time
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)


        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    train(opt)
