# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  :
@time   :21-7-22 下午7:36
@IDE    :PyCharm
@document   :process.py
"""
from bert4keras.tokenizers import load_vocab, Tokenizer

import json
import random

pretrain_model_path = '/wang/pretrain_model/'
bert_path = pretrain_model_path + 'albert_base/'
config_path = bert_path + 'albert_config.json'
checkpoint_path = bert_path + 'model.ckpt-best'
vocab_path = bert_path + 'vocab_chinese.txt'
domains = set()
intents = []
slots = []
sentences = []


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[unused1]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[UNK]')
            else:
                tokens.append(_ch)

    return tokens


intent_label2id, intent_id2label, slot_label2id, slot_id2label = json.load(
    open('./labels.json', 'r', encoding='utf-8')
)

# 加载并精简词表,建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

tokenizer = Tokenizer(token_dict)


def load_data():
    datas = []
    data = json.load(open('test_A_text.json', 'r', encoding='utf-8'))
    for k, d in data.items():
        datas.append(d)
    return datas


data = load_data()
random.shuffle(data)
num = len(data)*0.1
valid_data = data[:int(num)]
train_data = data[int(num):]

f_int = open('dialogue/test/label', 'w', encoding='utf-8')
f_seq_in = open('dialogue/test/seq.in', 'w', encoding='utf-8')
f_text_in = open('dialogue/test/text.in', 'w', encoding='utf-8')
f_seq_out = open('dialogue/test/seq.out', 'w', encoding='utf-8')
for item in train_data:
    intent = item.get('intent', 'UNK')
    # slot = list(item['slots'].keys())

    text = item['text']
    # token_ids, segment_ids = tokenizer.encode(first_text=text.lower())

    # y1 = domain_label2id.get(item['domain'])
    # y2 = intent_label2id.get(item['intent'], 'UNK')
    tokens = fine_grade_tokenize(text.lower(), tokenizer)
    # tokens = tokenizer.tokenize(text.lower())[1:-1]
    tokens0 = list(text.lower())
    assert len(tokens) == len(tokens0)
    print(tokens)
    tag_labels = ['O'] * len(tokens)  # 先排除[cls], [sep]的影响
    # for k, vs in item['slots'].items():
    #     if type(vs) == list:
    #         for v in vs:
    #             # v_tokens = tokenizer.tokenize(v)[1: -1]
    #             v_tokens = fine_grade_tokenize(v, tokenizer)
    #             len_v = len(v_tokens)
    #             for i in range(len(tokens) - len_v + 1):
    #                 if tokens[i: i + len_v] == v_tokens:
    #                     tag_labels[i] = 'B-' + k
    #                     # tag_labels[i] = slot_label2id.get('B-' + k)
    #                     # tag_labels[i + 1: i + len_v] = [slot_label2id.get('I-' + k)] * (len_v - 1)
    #                     tag_labels[i + 1: i + len_v] = ['I-' + k] * (len_v - 1)
    #                     break
    #     else:
    #         # v_tokens = tokenizer.tokenize(vs)[1: -1]
    #         v_tokens = fine_grade_tokenize(vs, tokenizer)
    #
    #         len_v = len(v_tokens)
    #         for i in range(len(tokens) - len_v + 1):
    #             if tokens[i: i + len_v] == v_tokens:
    #                 # tag_labels[i] = slot_label2id.get('B-' + k)
    #                 # tag_labels[i + 1: i + len_v] = [slot_label2id.get('I-' + k)] * (len_v - 1)
    #                 tag_labels[i] = 'B-' + k
    #                 tag_labels[i + 1: i + len_v] = ['I-' + k] * (len_v - 1)
    #                 break
    # print(tag_labels)
    f_int.write(intent+'\n')
    f_seq_in.write(' '.join(tokens)+'\n')
    f_text_in.write(' '.join(tokens0)+'\n')
    f_seq_out.write(' '.join(tag_labels)+'\n')
f_seq_in.close()
f_seq_out.close()
f_int.close()
f_text_in.close()
