from transformers import BertTokenizer, BertModel
from pprint import pprint
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
import csv

from utils import load_data, split_DevSet, dictionries, padding, collate_fn, modify_slot
from model3 import Lang, IntentsAndSlots
from functions import init_weights, train_loop, eval_loop


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# String to tokenize
text = "Hello, this is an example."

# Tokenize the text
tokenized_text = tokenizer.tokenize(text)
tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
decoded_sentence = tokenizer.decode(tokens)

# Print tokenized text
print(tokenizer(text))




tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
train_raw, dev_raw, y_test, y_dev, y_train = split_DevSet(tmp_train_raw, test_raw)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=False) # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased", force_download=False) # Download the model
train_raw1 = modify_slot(train_raw, tokenizer)
dev_raw1 = modify_slot(dev_raw, tokenizer)
test_raw1 = modify_slot(test_raw, tokenizer)


corpus = train_raw1 + dev_raw1 + test_raw1
intents = set([line['intent'] for line in corpus])
slots = set(sum([line['slots'].split() for line in corpus],[]))

lang = Lang(intents, slots,  cutoff=0)

train_raw1 = modify_slot(train_raw, tokenizer)
train_dataset = IntentsAndSlots(train_raw, tokenizer, lang)
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)


slots = set(sum([line['slots'].split() for line in corpus],[]))
print(slots)

print(len(lang.slot2id))

