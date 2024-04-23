#put all the functions needed to preprocess and load the dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0") 


'''
read the file and add <eos> - end of sentence - 
to indicate the end of a sentence or a sequence
'''
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output



'''
creates a vocabulary dictionary from a given corpus of text data, 
including optional special tokens
'''
def get_vocab(corpus, special_tokens=[]):
    #initialization of the dictionary
    output = {} 
    i = 0 
    #start from 0 and iterates over the list special_tokens and assign an index for all soecial_tokens
    for st in special_tokens:
        output[st] = i
        i += 1
    #start from i and iterate over each sentence in the corpus 
    for sentence in corpus:
        for w in sentence.split():
            if w not in output: #check if the word is not in the dictionary 
                output[w] = i #add the the word in the dictionary and assign an index  
                i += 1
    return output



def collate_fn(data, pad_token):

    '''
    trasformazione di un batch di sequenze in un tensore PyTorch, 
    dove tutte le sequenze sono "paddate" alla lunghezza massima del batch 
    per assicurare che tutti i tensori abbiano la stessa dimensione.

    '''
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences] #calcola lunghezza di ogni sequenza (frase) nel batch
        max_len = 1 if max(lengths)==0 else max(lengths) #cerca la lunghezza massisma
        #crea un tensore di lunghezza max_leb, usa pad_token per fillare gli spazi vuoti
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        #il tensore risultante viene scollegato dal grafo computazionale di PyTorch per evitare che il backpropagation venga eseguito su queste operazioni
        padded_seqs = padded_seqs.detach()  
        return padded_seqs, lengths
    

    '''
    Ordina il batch dalla sequanza più lunga a quella più corta 
    utile per certi tipi di reti neurali come le LSTM, dove può essere vantaggioso processare prima le sequenze più lunghe

    Poi crea un nuovo dizionario che è il nuovo batch e li sposta nella GPU
    '''
    #ordina il batch in base alla lunghezza delle sequenze
    data.sort(key=lambda x: len(x["source"]), reverse=True) 

    #prepara il nuovo batch 
    new_item = {}
    #crea nuvoo dizionario in cui per ogni chiave raccoglie i dati corrispondenti in liste separate 
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    #sposta nella GPU
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    #numero totale di token nel batch (può essere utilizzato per normalizzare la loss durante l'addestramento)
    new_item["number_tokens"] = sum(lengths)
    return new_item
