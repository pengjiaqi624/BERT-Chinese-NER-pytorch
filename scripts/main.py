#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 12:35
# @Author  : JJkinging
# @File    : main.py

import os
import pickle
import torch
import json
from utils import load_vocab, read_corpus
from config import Config
import matplotlib.pyplot as plt
import torch.nn as nn

import sys
sys.path.append('E:\\Program\\text-NER\\SCU-JJkinging_BERT-Chinese-NER-pytorch-master\\model')    # 跳到上级目录下面（sys.path添加目录时注意是在windows还是在Linux下，windows下需要‘\\'否则会出错。）
print(sys.path)
from BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from optimization import BertAdam
from torch.utils.data import DataLoader, TensorDataset
from utils import train, dev
#import torchsnooper
import sys
import os
import time
# @torchsnooper.snoop()
def main():
    device = torch.device("cpu")    #"cuda:0" if torch.cuda.is_available() else
    config = Config()
    print('Loading corpus...')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    print('Train dataset loaded.')
    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)
    print('Dev dataset loaded.')
    test_ids = torch.LongTensor([temp.input_id for temp in test_data])
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
    test_tags = torch.LongTensor([temp.label_id for temp in test_data])

    test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)
    print('Test dataset loaded.')

    model = BERT_BiLSTM_CRF(tagset_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)
    #设置模型待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.lr,
                         warmup=config.warmup,   #warmup->让学习率先增大，后减小?0.1？
                         t_total=len(train_loader) * config.epochs)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode="max",
    #                                                        factor=0.5,
    #                                                        patience=1)


    # Continuing training from a checkpoint if one was given as argument.
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # -------------------- Training epochs ------------------- #
    train(model,train_loader, dev_loader, test_loader,optimizer)
    # 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == "__main__":

    # 自定义目录存放日志文件
    log_path = '../Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    main()






