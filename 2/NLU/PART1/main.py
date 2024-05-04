#write the calls to the functions needed to output the results asked by the exercise
import os
import torch
from collections import Counter
from pprint import pprint

from utils import load_data, split_DevSet, dictionries
from model import Lang, IntentsAndSlots


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
CRATE DICTIONARIES FOR MAPPING
'''
print('='*89)
print('MAPPING')
w2id, slot2id, intent2id = dictionries(train_raw, dev_raw, test_raw)
sent = 'I wanna a flight from Toronto to Kuala Lumpur'
mapping = [w2id[w] if w in w2id else w2id['unk'] for w in sent.split()] #se la parola non c'è nel dizionario, mette 1
print(sent, mapping)
print('# Vocab:', len(w2id)-2) # we remove pad and unk from the count
print('# Slots:', len(slot2id)-1)
print('# Intent:', len(intent2id))


words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw # We do not wat unk labels,
                                        # however this depends on the research purpose
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])
lang = Lang(words, intents, slots, cutoff=0)


'''
CRATE THE DATASETS
'''
train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)