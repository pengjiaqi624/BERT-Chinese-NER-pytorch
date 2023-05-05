# -*-coding = uft-8 -*-
# @Time : 2022/9/20 16:13
# @Author : PENG
# @File : seqeval_try.py
# @Software : PyCharm
# -*- coding: utf-8 -*-
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from seqeval.metrics import performance_measure
y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'B-PER', 'I-PER', 'I-PER', 'I-PER']]
# y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
# y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
print("accuary: ", accuracy_score(y_true, y_pred))
print("p: ", precision_score(y_true, y_pred))
print("r: ", recall_score(y_true, y_pred))
print("f1: ", f1_score(y_true, y_pred))
print("classification report: ")
print(classification_report(y_true, y_pred))
print("performance_measure: ")
print(performance_measure(y_true, y_pred))
