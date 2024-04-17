# coding=utf-8
from model import *

train_text, train_label, train_rel, train_pos = read_data("data/train_bieos.txt", 'data/train_rel.txt')
test_text, test_label, test_rel, test_pos = read_data("data/test_bieos.txt", 'data/test_rel.txt')
label_2_index, index_2_label = build_label(train_label)
rel_2_index, index_2_rel = build_rel(train_rel)
tokenizer = BertTokenizer.from_pretrained("../bert_base_chinese")

test_dataset = BertDataset(test_text, test_label,test_rel,test_pos,label_2_index,rel_2_index, tokenizer, 0)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

ner_model = torch.load("ner_model.pt")
re_model = torch.load('re_model.pt')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

all_pre = []
all_tag = []
test_out = []
do_input = True
if do_input:
    text = input("输入: ")
    text=[i for i in text]
    text_index = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    text_index = text_index.to(device)

    pre = ner_model.forward(text_index)
    pre = pre[0][1:-1]
    pre = [index_2_label[i] for i in pre]
    print(" ".join(f"{w}:{t}" for w, t in zip(text, pre)))
    entitys, pos = find_entity(pre, text)
    if len(entitys) < 2:
        print("检测到的实体少于两个")
    else:
        for i in range(len(entitys)):
            for j in range(i+1,len(entitys)):
                p=[]
                p.extend(pos[i])
                p.extend(pos[j])
                re_pre=re_model.forward(text_index,[p])
                print(f'({entitys[i]},{entitys[j]},{index_2_rel[re_pre[0]]})')
else:
    for id, (batch_text_index, batch_label_index, batch_rel_index, batch_pos, label_len) in enumerate(test_dataloader):
        text = test_text[id]

        batch_text_index = batch_text_index.to(device)
        batch_label_index = batch_label_index.to(device)
        pre = ner_model.forward(batch_text_index)

        tag = batch_label_index.cpu().numpy().tolist()

        p = pre[0][1:-1]
        t = tag[0][1:-1]

        p = [index_2_label[i] for i in p]
        t = [index_2_label[i] for i in t]

        all_pre.append(p)
        all_tag.append(t)

        print(id, '/', len(test_text))
    precision=precision_score(all_tag,all_pre)
    score = f1_score(all_tag, all_pre)
    print(f'pre:{precision}')
    print(f"score:{score:.2f}")
