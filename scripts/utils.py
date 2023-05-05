#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 22:58
# @Author  : JJkinging
# @File    : utils.py
import torch
import os
import time
from datetime import timedelta
import torch.nn as nn
from tqdm import tqdm
from estimate import Precision, Recall, F1_score
from config import Config
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,performance_measure
'''工具类函数'''

class InputFeature(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

def load_vocab(vocab_file):
    '''construct word2id or label2id'''
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab

def read_corpus(path, max_length, label_dic, vocab):
    '''
    :param path: 数据集文件路径
    :param max_length: 句子最大长度
    :param label_dic: 标签字典
    :param vocab:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as fp:
        result = []
        words = []
        labels = []
        for line in tqdm(fp):
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[1])
            else:
                if len(contends) == 0 and len(words) > 0:
                    if len(words) > max_length - 2:
                        words = words[0:(max_length-2)]
                        labels = labels[0:(max_length-2)]
                    words = ['[CLS]'] + words + ['[SEP]']
                    labels = ['<START>'] + labels + ['<EOS>']
                    input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in words]
                    label_ids = [label_dic[i] for i in labels]
                    input_mask = [1] * len(input_ids)
                    # 填充
                    if len(input_ids) < max_length:
                        input_ids.extend([0]*(max_length-len(input_ids)))
                        label_ids.extend([0]*(max_length-len(label_ids)))
                        input_mask.extend([0]*(max_length-len(input_mask)))
                    assert len(input_ids) == max_length
                    assert len(label_ids) == max_length
                    assert len(input_mask) == max_length
                    feature = InputFeature(input_id=input_ids, label_id=label_ids, input_mask=input_mask)
                    result.append(feature)
                    # 还原words、labels = []
                    words = []
                    labels = []
        return result

def train(model,
          train_loader, dev_loader, test_loader,
          optimizer,
          ):
    '''
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    :param model: A torch module that must be trained on some input data.
    :param dataloader: A DataLoader object to iterate over the training data.
    :param optimizer: A torch optimizer to use for training on the input model.
    :param criterion: A loss criterion to use for training.
    :param max_gradient_norm: Max norm for gradient norm clipping.
    :return:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    '''
    config = Config()
    device = torch.device("cpu")    #"cuda:0" if torch.cuda.is_available() else
    print("\n",
          20 * "-",
          "Training BERT_BiLSTM_CRF model on device: {}".format(device),
          20 * "-")
    label_dic = load_vocab(config.label_file)
    id2tag = {label_dic[tag]: tag for tag in
              label_dic.keys()}  # {0: '<pad>', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG', 7: 'O', 8: '<START>', 9: '<EOS>'}
    model.train() # Switch the model to train mode.
    start =time.time()
    step = 0
    last_improve = 0 # 记录上一次更新的step值
    Train_loss = []
    Train_acc = []
    Dev_loss = []
    Dev_acc = []
    dev_best_loss = float('inf')    #初始化在验证集上的loss为inf（无穷大?）
    flag = False
    for epoch in range(config.epochs):
        print(f'Epoch [{epoch + 1}/{config.epochs}]')
        for i, batch in enumerate(train_loader):
            step += 1
            model.zero_grad()
            inputs, masks, tags = batch #list[句子，掩码，标签];size（batchsize，padsize）
            inputs = inputs.to(device)
            masks = masks.byte().to(device)
            tags = tags.to(device)
            feats = model(inputs, masks)#torch.Size([batchsize，padsize, tagsize])
            loss = model.loss(feats, tags, masks)
            loss.backward()
            optimizer.step()
            # tmp = []
            # pre_output = []
            # true_output = []    #batchsize大小的list，嵌套列表内容为tag
            # real_length = torch.sum(masks, dim=1)   #batchsize个长度的list，内容为masks和
            # j = 0
            # for line in tags.numpy().tolist():  #遍历tag（batchsize，padsize）的每一行
            #     tmp.append(line[0: real_length[j]]) #取每一行的real_length长度出来，
            #     j += 1
            # true_output.append(tmp)
            # out_path = model.predict(feats, masks)
            # pre_output.append(out_path)
            # '''
            # debug
            # '''
            # tags = [(1, 2), (3, 4), (5, 6)]
            # results = {}
            # num3 = 1
            # for tag in tags:
            #
            #     B_label_id = tag[0]
            #     I_label_id = tag[1]
            #     result = []
            #     batch_tag = []
            #     sentence_tag = []
            #     for batch in pre_output:
            #         for out_id_list in batch:
            #             for num in range(len(out_id_list)):
            #                 if out_id_list[num] == B_label_id:
            #                     start_pos = num
            #                 if out_id_list[num] == I_label_id and out_id_list[num - 1] == B_label_id:
            #                     length = 2
            #                     start_pos = num - 1
            #                     for num2 in range(num, len(out_id_list)):
            #                         if out_id_list[num2] == I_label_id and out_id_list[num2 - 1] == I_label_id:
            #                             length += 1
            #                             if out_id_list[num2] == 9:  # 到达末尾
            #                                 sentence_tag.append((start_pos, length))
            #                                 break
            #                         if out_id_list[num2] == 7:
            #                             sentence_tag.append((start_pos, length))
            #                             break
            #             batch_tag.append(sentence_tag)
            #             sentence_tag = []
            #         result.append(batch_tag)
            #         batch_tag = []
            #     res = result
            #     results[num3] = res
            #     num3 += 1
            # pre = []
            # pre_result = results
            # for num in pre_result:  # 遍历PER LOC ORG
            #     for i, batch in enumerate(pre_result[num]):
            #         for j, seq_path_id in enumerate(batch):
            #             if len(seq_path_id) != 0:
            #                 for one_tuple in seq_path_id:
            #                     if one_tuple:
            #                         if pre_output[i][j][one_tuple[0]: one_tuple[0] + one_tuple[1]] == \
            #                                 true_output[i][j][one_tuple[0]: one_tuple[0] + one_tuple[1]]:
            #                             pre.append(1)
            #                         else:
            #                             pre.append(0)
            # if len(pre) != 0:
            #     return sum(pre) / len(pre)
            # else:
            #     return 0
            pre_output = []
            true_output = []
            out_path = model.predict(feats, masks)
            pre_output.extend([t for t in out_path])
            true_output.extend([[x for x in t.tolist() if x != 0] for t in tags])
            if step % 1 == 0:#6
                # 每多少轮输出在训练集和验证集上的效果
                train_true = [[id2tag[y] for y in x] for x in true_output]
                train_pred = [[id2tag[y] for y in x] for x in pre_output]
                train_acc = accuracy_score(train_true, train_pred)
                print(train_acc)
                dev_acc, dev_loss = dev(model, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = step
                    torch.save(model.state_dict(), os.path.join(config.target_dir, "BERT.pth"))
                else:
                    improve = ''
                end = time.time()
                time_dif = timedelta(seconds=int(round(end - start)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Dev Loss: {3:>5.2},  Dev Acc: {4:>6.2%},  Time: {5} {6}'
                #{位置，格式}
                print(msg.format(step, loss.item(), train_acc, dev_loss, dev_acc, time_dif,
                                 improve))  # .item() 精度比直接取值要高。另外还可用于遍历中。
                model.train()

            if step - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

        Train_loss.append(loss.item())
        Train_acc.append(train_acc)
        Dev_loss.append(dev_loss)
        Dev_acc.append(dev_acc)
    print('Train_loss:', Train_loss)
    print('Train_acc:', Train_acc)
    print('Dev_loss:', Dev_loss)
    print('Dev_acc:', Dev_acc)

    test(model, test_loader)


def dev(model,
          dev_loader,test= False,):
    model.eval()
    dev_loss = 0
    device = model.device
    pre_output = []
    true_output = []

    config = Config()
    label_dic = load_vocab(config.label_file)
    id2tag = {label_dic[tag]: tag for tag in
              label_dic.keys()}  # {0: '<pad>', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG', 7: 'O', 8: '<START>', 9: '<EOS>'}

    with torch.no_grad():

        for _, batch in enumerate(dev_loader):
            inputs, masks, tags = batch

            # real_length = torch.sum(masks, dim=1)
            # tmp = []
            # i = 0
            # for line in tags.numpy().tolist():
            #     tmp.append(line[0: real_length[i]])
            #     i += 1
            #
            # true_output.append(tmp)

            inputs = inputs.to(device)
            masks = masks.byte().to(device)
            tags = tags.to(device)

            feats = model(inputs, masks)
            loss = model.loss(feats, tags, masks)

            out_path = model.predict(feats, masks)
            dev_loss += loss.item()
            pre_output.extend([t for t in out_path])

            true_output.extend([[x for x in t.tolist() if x != 0] for t in tags])
    ave_loss = dev_loss / len(dev_loader)
    true = [[id2tag[y] for y in x] for x in true_output]
    pred = [[id2tag[y] for y in x] for x in pre_output]
    accuracy = accuracy_score(true, pred)
    if test:
        f1 = f1_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        report = classification_report(true, pred)
        return accuracy, precision, recall, f1, ave_loss, report
    model.train()
    return accuracy, ave_loss

    # # 计算准确率、召回率、F1值
    # precision = Precision(pre_output, true_output)
    # recall = Recall(pre_output, true_output)
    # f1_score = F1_score(precision, recall)
    # # accuracy = metrics.accuracy_score(true_output, pre_output)
    # estimator = (precision, recall, f1_score)
    # accuracy = 1
    # if test:
    #     report = 'hhh'
    #     # report = metrics.classification_report(true_output, pre_output)
    #     return epoch_loss, accuracy, precision, recall, f1_score, report
    # model.train()
    # return accuracy, epoch_loss

def test(model,test_loader):

    model.eval()
    accuracy, precision, recall, f1, loss, report = dev(model=model, dev_loader=test_loader,
                                                      test=True)
    # epoch_loss,accuracy, precision, recall, f1_score, report = valid(model,dataloader, test=True)
    msg1 = 'Test Loss:{0:5.2}, Test Acc:{1:6.2%}'
    print(msg1.format(loss, accuracy))
    msg2 = 'Test Precision:{}, Test Recall:{}'
    print(msg2.format(precision, recall,f1))
    print('Classification Report:')
    print(report)