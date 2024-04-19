# encoding=utf-8
from model import *

if __name__ == "__main__":
    # 读BIOS
    train_text, train_label, train_rel, train_pos = read_data("data/train_bieos.txt", 'data/train_rel.txt')
    dev_text, dev_label, dev_rel, dev_pos = read_data("data/dev_bieos.txt", 'data/dev_rel.txt')

    label_2_index, index_2_label = build_label(train_label)
    rel_2_index, index_2_rel = build_rel(train_rel)
    tokenizer = BertTokenizer.from_pretrained("../bert_base_chinese")

    print(rel_2_index)

    batch_size = 16
    epoch = 5
    max_len = 200
    lr = 0.00005
    lstm_hidden = 128
    num = int(len(train_label) / batch_size)
    count = 1

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BertDataset(train_text, train_label, train_rel, train_pos, label_2_index, rel_2_index, tokenizer,
                                max_len, False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BertDataset(dev_text, dev_label, dev_label, dev_pos, label_2_index, rel_2_index, tokenizer, max_len,
                              False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = BertReModel(len(rel_2_index)).to(device)
    opt = AdamW(model.parameters(), lr)

    best_score = -1
    for e in range(epoch):
        print('轮次：', e, "/", epoch)
        count = 0
        model.train()
        for batch_text_index, batch_label_index, batch_rel_index, batch_pos, label_len in train_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            batch_rel_index = batch_rel_index.to(device)
            loss = model.forward(batch_text_index, batch_pos, batch_rel_index)
            loss.backward()

            opt.step()
            opt.zero_grad()
            count += 1
            print(f"loss:{loss:.2f}  {count}/{num}")


        model.eval()
        all_entity_pre = []
        all_rel_tag = []
        all_rel_pre = []
        for batch_text_index, batch_label_index, batch_rel_index, batch_pos, label_len in train_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            batch_rel_index = batch_rel_index.to(device)
            pre_re = model.forward(batch_text_index, batch_pos)

            pre_re = pre_re.cpu().numpy().tolist()
            rel_tag = torch.argmax(batch_rel_index.cpu(), -1)
            rel_tag = rel_tag.numpy().tolist()

            all_rel_tag.extend(rel_tag)
            all_rel_pre.extend(pre_re)


        ac = accuracy_score(all_rel_tag, all_rel_pre)
        print(f'accuracy_score:{ac}')
        score = f1_score(all_rel_tag, all_rel_pre, average='macro')
        if score > best_score:
            torch.save(model, "re_model.pt")
            best_score = score

        print(f"best_score:{best_score:.2f},score:{score:.2f}")
