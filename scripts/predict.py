#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 18:35
# @Author  : JJkinging
# @File    : predict.py
import sys
import torch
sys.path.append('E:\\Program\\text-NER\\SCU-JJkinging_BERT-Chinese-NER-pytorch-master\\model')    # 跳到上级目录下面（sys.path添加目录时注意是在windows还是在Linux下，windows下需要‘\\'否则会出错。）
print(sys.path)
from BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from config import Config
from utils import load_vocab
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm

'''用于识别输入的句子（可以换成批量输入）的命名实体
    <pad>   0
    B-Location   1
    I-Location   2
    B-Time   3
    I-Time   4
    O       7
    <START> 8
    <EOS>   9
'''
# tags = [(1, 2), (3, 4)]
tags = [(1, 2), (3, 4), (5, 6)]

def predict(input_seq, max_length=256):
    '''
    :param input_seq: 输入一句话
    :return:
    '''
    config = Config()
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tagset_size = len(label_dic)
    device = torch.device("cpu")    #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BERT_BiLSTM_CRF(tagset_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)

    checkpoint = torch.load(config.target_dir+'/BERT.pth')#RoBERTa_best.pthBERT.pth
    model.load_state_dict(checkpoint)#["model"]

    # 构造输入
    input_list = []
    for i in range(len(input_seq)):
        input_list.append(input_seq[i])

    if len(input_list) > max_length - 2:
        input_list = input_list[0:(max_length - 2)]
    input_list = ['[CLS]'] + input_list + ['[SEP]']

    input_ids = [int(vocab[word]) if word in vocab else int(vocab['[UNK]']) for word in input_list]
    input_mask = [1] * len(input_ids)

    if len(input_ids) < max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))
        input_mask.extend([0] * (max_length - len(input_mask)))
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    # 变为tensor并放到GPU上, 二维, 这里mask在CRF中必须为unit8类型或者bool类型
    input_ids = torch.LongTensor([input_ids]).to(device)
    input_mask = torch.ByteTensor([input_mask]).to(device)
    model.eval()
    with torch.no_grad():
        feats = model(input_ids, input_mask)
    # out_path是一条预测路径（数字列表）, [1:-1]表示去掉一头一尾, <START>和<EOS>标志
    #     out_path = model.crf.decode(feats, input_mask)[0][1:-1]
    out_path = model.predict(feats, input_mask)[0][1:-1]
    res = find_all_tag(out_path)

    Level = []
    Location = []
    Time = []
    for name in res:
        if name == 1:
            for i in res[name]:
                Location.append(input_seq[i[0]:(i[0]+i[1])])
        if name == 2:
            for j in res[name]:
                Time.append(input_seq[j[0]:(j[0]+j[1])])
        if name == 3:
            for k in res[name]:
                Level.append(input_seq[k[0]:(k[0] + k[1])])
    Locations.append(Location)#.extend()会以元素添加
    Times.append(Time)
    Levels.append(Level)
    # 输出结果
    print('预测结果:','\n', 'Location:', Location, '\n', 'Time:', Time, '\n', 'Level:', Levels)
    print('预测结果:', '\n', 'Location:', Locations, '\n', 'Time:', Times)
    print('预测结果:','\n', 'Location:', Location)
    print('预测结果:', '\n', 'Location:', Locations)

def find_tag(out_path, B_label_id=1, I_label_id=2):
    '''
    找到指定的label
    :param out_path: 模型预测输出的路径 shape = [1, rel_seq_len]
    :param B_label_id:
    :param I_label_id:
    :return:
    '''
    sentence_tag = []
    for num in range(len(out_path)):
        if out_path[num] == B_label_id:
            start_pos = num
        if out_path[num] == I_label_id and out_path[num-1] == B_label_id:
            length = 2
            for num2 in range(num, len(out_path)):
                if out_path[num2] == I_label_id and out_path[num2-1] == I_label_id:
                    length += 1
                    if num2 == len(out_path)-1:  # 如果已经到达了句子末尾
                        sentence_tag.append((start_pos, length))
                        return sentence_tag
                if out_path[num2] == 7:
                    sentence_tag.append((start_pos, length))
                    break
    return sentence_tag

def find_all_tag(out_path):
    num = 1  # 1: PER、 2: LOC、3: ORG
    result = {}
    for tag in tags:
        res = find_tag(out_path, B_label_id=tag[0], I_label_id=tag[1])
        result[num] = res
        num += 1
    return result

conn = pymysql.connect(host="127.0.0.1",
                       port=3306,
                       user="root",
                       password="pengjiaqi",
                       db="zzweibo2",
                       charset="utf8")
def load_data_from_mysql(table):
    '''
    :param table: 查询表名
    :return: 表数据
    '''
    sql = "SELECT * FROM "+ str(table)
    # cursor = conn.cursor()
    # cursor.execute(sql)
    # result = cursor.fetchall()
    data_frame = pd.read_sql(sql, conn)
    return data_frame

if __name__ == "__main__":
    engine = create_engine("mysql+pymysql://root:pengjiaqi@127.0.0.1:3306/zzweibo2")#https://zhuanlan.zhihu.com/p/364688931
    
    input = load_data_from_mysql('positive')
    # input = input[0:10]

    Locations = []
    Times = []
    Levels = []
    for i in tqdm(range(input.shape[0])):
        input_seq = str(input.iloc[i,6])
        print(input_seq)
        predict(input_seq)

    input.insert(input.shape[1], 'NER_Location', Locations) #不能为-1
    input.insert(input.shape[1], 'NER_Time', Times)
    input.insert(input.shape[1], 'NER_Level', Levels)
    for i in tqdm(range(input.shape[0])):
        input.iloc[i,-1]= str(input.iloc[i,-1]).strip('[').strip(']').replace("'","")
        input.iloc[i,-2]= str(input.iloc[i,-2]).strip('[').strip(']').replace("'","")
        input.iloc[i, -3] = str(input.iloc[i, -3]).strip('[').strip(']').replace("'", "")
    print(input)
    input.to_sql('positive_NER',engine, if_exists="append", index=False)#pd 的 to_sql 不能使用 pymysql 的连接，否则就会直接报错

