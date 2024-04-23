import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader


class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    
'''
PennTreeBank è una sottoclasse di torch.utils.data.Dataset, 
che è una classe di PyTorch utilizzata per rappresentare set di dati personalizzati 
per l'addestramento dei modelli di deep learning.
'''

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = [] #lista di parole di partenza
        self.target = [] #lista di parole di destinazione di ciascuna frase
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    #numero di frasi nel corpus
    def __len__(self):
        return len(self.source)

    '''
    recupera un elemento specifico dal set di dati. In particolare:
    per un indice idx specifico, converte le sequenze di parole corrispondenti in tensori e
    restituisce un dizionario contenente il tensore della sorgente ('source') e il tensore 
    di destinazione ('target') come campione.
    '''
    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx]) #Rappresenta un tensore di interi a 64 bit, è un tensore di long (il long è un intero a 64bit)
        trg = torch.LongTensor(self.target_ids[idx]) #Rappresenta un tensore di interi a 64 bit 
        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    #mappa le sequenze di token ai corrispondenti identificatori di parole utilizzando la mappatura definita nel parametro lang
    def mapping_seq(self, data, lang): 
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res