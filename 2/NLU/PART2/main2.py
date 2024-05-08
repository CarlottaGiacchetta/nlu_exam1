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

from utils import load_data, split_DevSet, dictionries, padding, collate_fn
from model import Lang, IntentsAndSlots,  ModelIAS, ModelIAS_Bidirectional, ModelIAS_Bidirectional_drop
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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=True) # Download the tokenizer




model = BertModel.from_pretrained("bert-base-uncased", force_download=True) # Download the model

inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
pprint(inputs)