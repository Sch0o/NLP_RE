# coding=utf-8
import re

import pandas as pd

class_label = [['B-SIN', 'I-SIN', 'E-SIN', 'S-SIN'], ['B-NAME', 'I-NAME', 'E-NAME', 'S-NAME'],
               ['B-INS', 'I-INS', 'E-INS', 'S-INS']]


def swap(e1, e2):
    return e2, e1


def add_label(new_sents, label, site, class_num):
    for i in range(len(new_sents)):
        for n in site:
            if n == '':
                continue
            pos = find_all_sybstrings(new_sents[i], n)
            for p in pos:
                if len(n) == 1:
                    label[i][p] = class_label[class_num][3]
                else:
                    label[i][p:p + len(n)] = [class_label[class_num][1]] * len(n)
                    label[i][p] = class_label[class_num][0]
                    label[i][p + len(n) - 1] = class_label[class_num][2]
    return label


def split_excel():
    df = pd.read_excel('data/wenshu.xls')

    shuffled_df = df.sample(frac=1)

    total_rows = len(df)
    p1_rows = int(total_rows * 0.6)
    p2_rows = int(total_rows * 0.8)
    p3_rows = total_rows

    df_train = shuffled_df.iloc[0:p1_rows]
    df_dev = shuffled_df.iloc[p1_rows:p2_rows]
    df_test = shuffled_df.iloc[p2_rows:p3_rows]
    df_train = df_train.loc[:, ['案由', '公诉机关', '当事人', '正文']]
    df_dev = df_dev.loc[:, ['案由', '公诉机关', '当事人', '正文']]
    df_test = df_test.loc[:, ['案由', '公诉机关', '当事人', '正文']]

    df_train.to_excel('data/train.xlsx', index='False')
    df_test.to_excel('data/test.xlsx', index='False')
    df_dev.to_excel('data/dev.xlsx', index='False')


def find_all_sybstrings(string, sub):
    all_position = []
    position = string.find(sub)
    while position != -1:
        all_position.append(position)
        position = string.find(sub, position + 1)
    return all_position


def add_rel(entitys, sents, labels):
    rel_label = ['公诉机关', '罪行', '无关']
    all_sents = []
    all_pos = []
    all_rel = []
    all_label = []
    for i in range(len(sents)):
        print(f'{i}/{len(sents)}')
        for k in range(0, 3):
            for j in range(k + 1, 3):
                for e1 in entitys[k]:
                    for e2 in entitys[j]:
                        pos1 = find_all_sybstrings(sents[i], e1)
                        pos2 = find_all_sybstrings(sents[i], e2)
                        if pos1 == [] or pos2 == [] or len(e1) == 0 or len(e2) == 0:
                            continue
                        if pos1[0] > pos2[0]:
                            pos1[0], pos2[0] = swap(pos1[0], pos2[0])
                            e1, e2 = swap(e1, e2)
                        if pos2[0] + len(e2) + 4 > 200:
                            continue
                        pos2[0] = pos2[0] + 2
                        sentence = sents[i]
                        sentence = sentence[0:pos1[0]] + "$" + sentence[pos1[0]:pos1[0] + len(e1)] \
                                   + "$" + sentence[pos1[0] + len(e1):]
                        pos1[0] = pos1[0] + 1
                        sentence = sentence[0:pos2[0]] + "#" + sentence[pos2[0]:pos2[0] + len(e2)] \
                                   + "#" + sentence[pos2[0] + len(e2):]
                        pos2[0] = pos2[0] + 1

                        all_sents.append(sentence)
                        all_label.append(labels[i])
                        all_pos.append([pos1[0], pos1[0] + len(e1) - 1, pos2[0], pos2[0] + len(e2) - 1])
                        if k == 0 and j == 1:
                            all_rel.append(rel_label[1])
                        elif k == 1 and j == 2:
                            all_rel.append(rel_label[0])
                        else:
                            all_rel.append(rel_label[2])
    return all_sents, all_label, all_rel, all_pos


def split_text(text):  # 正文分句
    sentences = re.split('(\?|\n|\u3000)', text)
    new_sents = []
    label = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        if len(sent) != 0:
            if sent[-1] == '\n' or '\u3000':
                sent = sent[:-1]
            if len(sent) != 0:
                new_sents.append(sent)
                label.append(['O'] * (len(sent) + 4))
    return new_sents, label


def excel_to_data(filename):
    all_text = []
    all_label = []
    all_rel = []
    all_pos = []
    df = pd.read_excel(filename)
    df = df.astype(str)
    for i in range(len(df)):
        if i > 5000:
            break
        df.iloc[i, 1] = df.iloc[i, 1][2:-2]
        sin = df.iloc[i, 1].split('、')

        ins = [df.iloc[i, 2]]

        name = df.iloc[i, 3]
        df.iloc[i, 3] = name[name.find('@') + 1:]
        name = df.iloc[i, 3].split(';')

        text = df.iloc[i, 4].replace(" ", "")

        new_sents, labels = split_text(text)
        entitys = [sin, name, ins]
        new_sents, labels, rels, pos, = add_rel(entitys, new_sents, labels)
        labels = add_label(new_sents, labels, sin, 0)
        labels = add_label(new_sents, labels, name, 1)
        labels = add_label(new_sents, labels, ins, 2)
        all_text += new_sents
        all_label += labels
        all_rel += rels
        all_pos += pos
    return all_text, all_label, all_rel, all_pos


def to_bioes(all_text, all_label, filename):  # 转成BIOS格式
    text_out = []
    for i in range(len(all_text)):
        if i >= 1000:
            break
        text_out.extend([f"{w} {t}" for w, t in zip(all_text[i], all_label[i])])
        text_out.append('')
        with open(filename, "w", encoding='utf-8') as f:
            f.write("\n".join(text_out))
        print(i, "/", len(all_text))


def to_txt(all_rel, all_pos, filename):
    text_out = [f"{r} {p[0]} {p[1]} {p[2]} {p[3]}" for r, p in zip(all_rel, all_pos)]
    text_out = text_out[:1000]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_out))


if __name__ == "__main__":
    is_split = False
    if is_split:
        split_excel()

    # all_text, all_label, all_rel, all_pos = excel_to_data('data/train.xlsx')
    # to_bioes(all_text, all_label, 'data/train_bieos.txt')
    # to_txt(all_rel, all_pos, 'data/train_rel.txt')
    all_text, all_label, all_rel, all_pos = excel_to_data('data/dev.xlsx')
    to_bioes(all_text, all_label, 'data/dev_bieos.txt')
    to_txt(all_rel, all_pos, 'data/dev_rel.txt')
    all_text, all_label, all_rel, all_pos = excel_to_data('data/test.xlsx')
    to_bioes(all_text, all_label, 'data/test_bieos.txt')
    to_txt(all_rel, all_pos, 'data/test_rel.txt')
