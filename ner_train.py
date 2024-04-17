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
    epoch = 10
    max_len =30
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

    model = BertNerModel(lstm_hidden, index_2_label).to(device)
    opt = AdamW(model.parameters(), lr)

    best_score = -1
    for e in range(epoch):
        print('轮次：', e, "/", epoch)
        count = 0
        model.train()
        for batch_text_index, batch_label_index, batch_rel_index, batch_pos, label_len in train_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            loss = model.forward(batch_text_index, batch_label_index)
            loss.backward()

            opt.step()
            opt.zero_grad()

            count += 1
            print(f"loss:{loss:.2f}  {count}/{num}")

        model.eval()
        all_pre = []
        all_tag = []
        for batch_text_index, batch_label_index, batch_rel_index, batch_pos, label_len in train_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model.forward(batch_text_index)

            tag = batch_label_index.cpu().numpy().tolist()

            for p, t, l in zip(pre, tag, label_len):
                p = p[1:1 + l]
                t = t[1:1 + l]

                p = [index_2_label[i] for i in p]
                t = [index_2_label[i] for i in t]

                all_pre.append(p)
                all_tag.append(t)
        score = f1_score(all_tag, all_pre)
        if score > best_score:
            torch.save(model, "ner_model.pt")
            best_score = score
        print(f"best_score:{best_score:.2f},score:{score:.2f}")
