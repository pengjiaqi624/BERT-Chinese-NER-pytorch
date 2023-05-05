# -*-coding = uft-8 -*-
# @Time : 2022/10/8 14:59
# @Author : PENG
# @File : prepare_data.py
# @Software : PyCharm
from sklearn.model_selection import train_test_split
import pandas as pd

data = []
with open (r"E:\Program\text-NER\YEDDA-master\demotext\label_all.txt.ann.txt","r",encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('"')
        data.append(line)
    print(data)

#划分数据集
def train_dev_test_split(df,train_size,test_size):
    train, middle = train_test_split(df,train_size=train_size)
    dev,test =train_test_split(middle,test_size=test_size)  #development
    return train,test,dev

train_list,test_list,dev_list = train_dev_test_split(data,0.8,0.5)
train = pd.DataFrame(train_list)#.strip('"')
test = pd.DataFrame(test_list)#.strip('"')
dev = pd.DataFrame(dev_list)#.strip('"')
train.to_csv(r"E:\Program\text-NER\YEDDA-master\demotext\train.txt", index=False, sep='\t',header = False,encoding = 'UTF-8')
test.to_csv(r"E:\Program\text-NER\YEDDA-master\demotext\test.txt", index=False, sep='\t',header = False,encoding = 'UTF-8')
dev.to_csv(r"E:\Program\text-NER\YEDDA-master\demotext\dev.txt", index=False, sep='\t',header = False,encoding = 'UTF-8')

