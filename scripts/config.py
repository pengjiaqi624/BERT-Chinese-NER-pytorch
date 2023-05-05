#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 12:40
# @Author  : JJkinging
# @File    : config.py
class Config(object):
    '''配置类'''

    def __init__(self):
        self.label_file = '../dataset/tag.txt'
        self.train_file = '../dataset/train.txt'
        self.dev_file = '../dataset/dev.txt'
        self.test_file = '../dataset/test.txt'
        self.vocab = '../dataset/bert/vocab.txt'
        # 可以换成RoBERTa的中文预训练模型（哈工大提供）
        self.pretrain_model_name = '../pretrained/bert'#RoBERTa_zh_L12_PyTorch,bert，chinese-roberta-wwm-ext
        self.use_cuda = False
        self.max_length = 128
        self.batch_size = 64 #64,16
        self.epochs = 20
        self.bert_embedding = 768
        self.rnn_hidden = 128
        self.rnn_layer = 1
        self.dropout = 0.5  #rnn 后的dropout层；rnn是否需要dropout？
        self.lr = 0.00002
        self.weight_decay = 0.01
        self.warmup = 0.05
        # self.checkpoint = '../result/checkpoints/RoBERTa_result'
        self.checkpoint =  None
        self.target_dir = '../result/checkpoints/BERT'#RoBERTa_result,BERT
        self.require_improvement = 1000
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


# if __name__ == '__main__':
#     con = Config()
#     con.update(gpu=8)
#     print(con.gpu)
#     print(con)