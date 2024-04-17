# encoding=utf-8
import pandas as pd

class_label = [['B-PEO', 'I-PEO', 'E-PEO', 'S-PEO']]


def split_excel():  # 分割数据集
    df = pd.read_excel('data/people.xlsx')

    shuffled_df = df.sample(frac=1)

    total_rows = len(df)
    p1_rows = int(total_rows * 0.6)
    p2_rows = int(total_rows * 0.8)
    p3_rows = total_rows

    df_train = shuffled_df.iloc[0:p1_rows]
    df_dev = shuffled_df.iloc[p1_rows:p2_rows]
    df_test = shuffled_df.iloc[p2_rows:p3_rows]
    return df_train, df_dev, df_test


def find_all_sybstrings(string, sub):
    all_position = []
    position = string.find(sub)
    while position != -1:
        all_position.append(position)
        position = string.find(sub, position + 1)
    return all_position


def add_label(df_data, class_num):
    all_label = []
    all_text = []
    all_rel = []
    all_pos = []
    for index, row in df_data.iterrows():
        have_name = 0
        entity_pos = []
        sentence = row[3].replace(" ", "")

        label = [0] * (len(sentence) + 4)
        name = [row.iloc[0], row.iloc[1]]
        name_index = 1
        for n in name:
            n = n.replace(" ", "")
            pos = find_all_sybstrings(sentence, n)
            if pos == [] or pos[0] + len(n) > 30:
                have_name = 1
                break
            # if name_index == 1:
            #     sentence = sentence[0:pos[0]] + "$" + sentence[pos[0]:pos[0] + len(n)] + "$" + sentence[
            #                                                                                    pos[0] + len(n):]
            #     name_index = 2
            #
            # else:
            #     sentence = sentence[0:pos[0]] + "#" + sentence[pos[0]:pos[0] + len(n)] + "#" + sentence[
            #                                                                                    pos[0] + len(n):]
            # p = pos[0] + 1
            p=pos[0]
            entity_pos.append(p)
            entity_pos.append(p + len(n) - 1)
            if len(n) == 1:
                label[p] = class_label[class_num][3]
            else:
                label[p:p + len(n)] = [class_label[class_num][1]] * len(n)
                label[p] = class_label[class_num][0]
                label[p + len(n) - 1] = class_label[class_num][2]

        if have_name == 1:
            continue
        all_text.append(sentence)
        all_rel.append(row.iloc[2])
        all_pos.append(entity_pos)
        all_label.append(label)
    return all_label, all_text, all_rel, all_pos


def to_bmes_file(all_text, all_label, filename):  # 转成BIOS格式
    text_out = []
    for i in range(len(all_text)):
        text_out.extend([f"{w} {t}" for w, t in zip(all_text[i], all_label[i])])
        text_out.append('')
        with open(filename, "w", encoding='utf-8') as f:
            f.write("\n".join(text_out))
        print(i, "/", len(all_text))


def to_txt(all_rel, all_pos, filename):
    text_out = [f"{r} {p[0]} {p[1]} {p[2]} {p[3]}" for r, p in zip(all_rel, all_pos)]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_out))


if __name__ == '__main__':
    df_train, df_dev, df_test = split_excel()

    all_label, all_text, all_rel, all_pos = add_label(df_train, 0)
    to_txt(all_rel, all_pos, 'data/train_rel.txt')
    to_bmes_file(all_text, all_label, 'data/train_bieos.txt')

    all_label, all_text, all_rel, all_pos = add_label(df_dev, 0)
    to_bmes_file(all_text, all_label, 'data/dev_bieos.txt')
    to_txt(all_rel, all_pos, 'data/dev_rel.txt')

    all_label, all_text, all_rel, all_pos = add_label(df_test, 0)
    to_bmes_file(all_text, all_label, 'data/test_bieos.txt')
    to_txt(all_rel, all_pos, 'data/test_rel.txt')
