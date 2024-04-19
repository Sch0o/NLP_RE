# -*- coding: utf-8 -*-
import numpy
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from transformers import BertModel
import torch
import torch.nn as nn
from transformers import AdamW
# from seqeval.metrics import accuracy_score
# from seqeval.metrics import f1_score
# from seqeval.metrics import precision_score
# from seqeval.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from torchcrf import CRF


def read_data(BIOS_file, rel_file):  # 读BIOS
    all_text = []
    all_label = []
    all_rel = []
    all_pos = []
    with open(BIOS_file, "r", encoding='utf-8') as f:
        all_data = f.read().split("\n")
        text = []
        label = []
        for data in all_data:
            if data == "":
                if text == []:
                    continue
                all_text.append(text)
                all_label.append(label)
                text = []
                label = []
            else:
                t, l = data.split(" ")
                text.append(t)
                label.append(l)
    with open(rel_file, "r", encoding='utf-8') as f:
        info = f.read().split('\n')
        for i in info:
            i = i.split(" ")
            all_rel.append(i[0])
            all_pos.append([int(p) for p in i[1:5]])
    print("read data done")
    return all_text, all_label, all_rel, all_pos


def build_label(train_label):  # 所有标签
    label_2_index = {"PAD": 0, "UNK": 1}
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    index_to_label=list(label_2_index)
    with open('label_list.txt','w')as file:
        file.write('\n'.join(index_to_label))
    return label_2_index, list(label_2_index)

def build_rel(train_rel):
    rel_2_index = {}
    for rel in train_rel:
        if rel not in rel_2_index:
            rel_2_index[rel] = len(rel_2_index)
    index_2_rel=list(rel_2_index)
    with open('rel_list.txt','w')as file:
        file.write('\n'.join(index_2_rel))
    return rel_2_index, list(rel_2_index)

def read_label_txt(file):
    index_to_label=[]
    with open(file,'r') as file:
        for line in file:
            index_to_label.append(line.strip())
    label_2_index={}
    for label in index_to_label:
        label_2_index[label]=len(label_2_index)

    return label_2_index,index_to_label


def to_onehot(label_2_index, all_rel):
    rel_index = [label_2_index[rel] for rel in all_rel]
    onehot = torch.nn.functional.one_hot(rel_index, len(label_2_index))
    return onehot





def find_entity(pre, sentence):
    entitys = []
    pos = []
    entity = ""
    for i in range(len(pre)):
        if 'B' == pre[i][0]:
            entity = entity + sentence[i]
        elif 'I' == pre[i][0] and entity != '':
            entity = entity + sentence[i]
        elif 'E' == pre[i][0] and entity != '':
            entity = entity + sentence[i]
            if entity not in entitys:
                entitys.append(entity)
                pos.append([i - len(entity) + 1, i])
                entity = ""
    return entitys, pos


class BertDataset(Dataset):
    def __init__(self, all_text, all_label, all_rel, all_pos, label_2_index, rel_2_index, tokenizer, max_len,
                 is_test=True):
        self.all_text = all_text
        self.all_label = all_label
        self.all_rel = all_rel
        self.all_pos = all_pos
        self.rel_2_index = rel_2_index
        self.label_2_index = label_2_index
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __getitem__(self, index):
        if self.is_test:
            self.max_len = len(self.all_text[index])
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]

        text_index = self.tokenizer.encode(text, add_special_token=True, max_length=self.max_len + 2,
                                           padding="max_length",
                                           truncation=True, return_tensor="pt")
        label_index = [0] + [self.label_2_index.get(l, 1) for l in label] + [0] + [0] * (self.max_len - len(label))
        rel_index = torch.tensor(self.rel_2_index[self.all_rel[index]])
        rel_index = torch.nn.functional.one_hot(rel_index, len(self.rel_2_index))
        pos = self.all_pos[index]
        text_index = torch.tensor(text_index)
        label_index = torch.tensor(label_index)
        pos = torch.tensor(pos)

        return text_index.reshape(-1), label_index, rel_index, pos, len(label)

    def __len__(self):
        return self.all_text.__len__()


class BertNerModel(nn.Module):
    def __init__(self, lstm_hidden, index_to_label):
        super().__init__()
        self.index_2_label = index_to_label
        self.class_num = len(index_to_label)
        self.bert = BertModel.from_pretrained("../bert_base_chinese")
        for name, param in self.bert.named_parameters():
            param.require_grad = False

        self.lstm = nn.LSTM(768, lstm_hidden, batch_first=True, num_layers=1, bidirectional=False)
        self.classifier_ner = nn.Linear(lstm_hidden, self.class_num)
        self.crf = CRF(self.class_num, batch_first=True)

    def forward(self, batch_text_index, batch_label_index=None):
        bert_out = self.bert(batch_text_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # 字符级别 篇章级别
        lstm_out, _ = self.lstm(bert_out0)
        pre_ner = self.classifier_ner(lstm_out)
        if batch_label_index is not None:
            loss_ner = -self.crf(pre_ner, batch_label_index)
            return loss_ner
        else:
            pre_ner = self.crf.decode(pre_ner)
            return pre_ner


class BertReModel(nn.Module):
    def __init__(self, rel_num):
        super().__init__()
        self.rel_num = rel_num
        self.bert = BertModel.from_pretrained("../bert_base_chinese")
        for name, param in self.bert.named_parameters():
            param.require_grad = False
        self.loss_fun = nn.CrossEntropyLoss()
        self.classifier_re1 = nn.Linear(768 * 3, 768)
        self.classifier_re2 = nn.Linear(768, self.rel_num)

    def forward(self, batch_text_index, batch_pos, batch_rel_index=None):
        bert_out = self.bert(batch_text_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # 字符级别 篇章级别
        batch_input = []
        bert_code = [bert_code.tolist() for bert_code in bert_out0]
        for i in range(len(batch_text_index)):
            input = []
            pos1 = [batch_pos[i][0] + 1, batch_pos[i][1] + 2]
            pos2 = [batch_pos[i][2] + 1, batch_pos[i][3] + 2]
            input.extend([sum(col) / (pos1[1] - pos1[0]) for col in zip(*bert_code[i][pos1[0]:pos1[1]])])
            input.extend(bert_code[i][0])
            input.extend([sum(col) / (pos2[1] - pos2[0]) for col in zip(*bert_code[i][pos2[0]:pos2[1]])])
            batch_input.append(input)
        batch_input = torch.tensor(batch_input)
        batch_input = batch_input.to(torch.device("cuda"))
        out = torch.relu(self.classifier_re1(batch_input))
        out = self.classifier_re2(out)

        if batch_rel_index is not None:
            loss_re = self.loss_fun(out, batch_rel_index.type("torch.cuda.FloatTensor"))
            loss = loss_re
            return loss
        else:
            pre_re = torch.argmax(out, -1)
            return pre_re
