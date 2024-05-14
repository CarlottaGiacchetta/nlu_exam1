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
from model3 import Lang, IntentsAndSlots, BertForJointIntentAndSlot
from functions import init_weights, train_loop, eval_loop


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0") # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0


'''
READ FILES
'''
tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
print('Train samples:', len(tmp_train_raw))
print('Test samples:', len(test_raw))

pprint(tmp_train_raw[0])

'''
CRATE A DEV SET 
'''

train_raw, dev_raw, y_test, y_dev, y_train = split_DevSet(tmp_train_raw, test_raw)
print('='*89)
print('INTENT DISTRIBUTIONS')
# Intent distributions
print('Train:')
pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
print('Dev:'), 
pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
print('Test:') 
pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
print('='*89)
# Dataset size
print('TRAIN size:', len(train_raw))
print('DEV size:', len(dev_raw))
print('TEST size:', len(test_raw))

'''
TOKENIZATION
'''

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=False) # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased", force_download=False) # Download the model
train_raw1 = modify_slot(train_raw, tokenizer)
dev_raw1 = modify_slot(dev_raw, tokenizer)
test_raw1 = modify_slot(test_raw, tokenizer)


corpus = train_raw1 + dev_raw1 + test_raw1
intents = set([line['intent'] for line in corpus])
slots = set(sum([line['slots'].split() for line in corpus],[]))
lang = Lang(intents, slots,  cutoff=0)

train_dataset = IntentsAndSlots(train_raw1, tokenizer, lang)
dev_dataset = IntentsAndSlots(dev_raw1, tokenizer, lang)
test_dataset = IntentsAndSlots(test_raw1, tokenizer, lang)
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)


 

'''
for sample in train_loader:
    print(sample)
    print('CONTROLLO UTTERANCES TOKEN')
    print("sample['utterances']",sample['utterances'])
    for token_ids in sample['utterances'].tolist():
        print(token_ids)
        print(tokenizer.decode(token_ids))

    print('CONTROLLO ATTENTION')
    print("sample['attention']",sample['attention'])
    for attention in sample['attention']:
        print(attention)
    

    
    
    print('CONTROLLO SLOT TOKEN')
    print("sample['y_slots']", sample['y_slots'])

    
    for token_ids in sample['y_slots'].tolist():
        print(token_ids)
        print(tokenizer.decode(token_ids))
    
'''

'''
FINE TUNING
'''
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

modellooo = BertForJointIntentAndSlot(model, num_intents = len(lang.intent2id), num_slots = len(lang.slot2id)).to(device)


patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
for x in tqdm(range(1, 20)):
    loss = train_loop(train_loader, optimizer, criterion_slots,
                                        criterion_intents, modellooo, clip = 5)
    if x % 5 == 0: # We check the performance every 5 epochs
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                      criterion_intents, modellooo, lang)
        losses_dev.append(np.asarray(loss_dev).mean())

        f1 = results_dev['total']['f']
        # For decreasing the patience you can also use the average between slot f1 and intent accuracy
        if f1 > best_f1:
            best_f1 = f1
            # Here you should save the model
            patience = 3
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                            criterion_intents, modellooo, lang)
print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])
   
    
    
   

