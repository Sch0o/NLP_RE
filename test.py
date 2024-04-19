# coding=utf-8
from model import *

test_text, test_label, test_rel, test_pos = read_data("data/test_bieos.txt", 'data/test_rel.txt')
label_2_index, index_2_label = read_label_txt('label_list.txt')
rel_2_index, index_2_rel = read_label_txt('rel_list.txt')
tokenizer = BertTokenizer.from_pretrained("../bert_base_chinese")

test_dataset = BertDataset(test_text, test_label,test_rel,test_pos,label_2_index,rel_2_index, tokenizer, 200,False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

ner_model = torch.load("ner_model.pt")
re_model = torch.load('re_model.pt')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

all_pre = []
all_tag = []
test_out = []
do_input = False
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
    re_model.eval()
    all_entity_pre = []
    all_rel_tag = []
    all_rel_pre = []
    for id, (batch_text_index, batch_label_index, batch_rel_index, batch_pos, label_len) in enumerate(test_dataloader):
        batch_text_index = batch_text_index.to(device)
        batch_label_index = batch_label_index.to(device)
        batch_rel_index = batch_rel_index.to(device)
        pre_re = re_model.forward(batch_text_index, batch_pos)

        pre_re = pre_re.cpu().numpy().tolist()
        rel_tag = torch.argmax(batch_rel_index.cpu(), -1)
        rel_tag = rel_tag.numpy().tolist()

        all_rel_tag.extend(rel_tag)
        all_rel_pre.extend(pre_re)
        print(f"{id}/{len(test_text)/16}")

    ac = accuracy_score(all_rel_tag, all_rel_pre)
    print(f'accuracy_score:{ac}')
    score = f1_score(all_rel_tag, all_rel_pre, average='macro')
    print(f"f1_score:{score:.2f}")
