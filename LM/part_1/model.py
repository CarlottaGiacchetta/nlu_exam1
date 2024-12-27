#the class of the model defined in PyTorch.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader 
import torch.optim as optim
from torch.autograd import Variable


'CLASSI GIà IMPLEMENTATE PER IL LOAD DEI DATI'


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
                    print('You have to deal with that') 
                    break
            res.append(tmp_seq)
        return res


class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    
    '''
    Questo modello accetta una sequenza di token come input, li trasforma in vettori di embedding, 
    li passa attraverso una RNN e la RNN restituisce due valori: l'output della RNN (rnn_out) e lo stato nascosto finale,
    ma qui ci interessa solo rnn_out. L'output della RNN viene poi proiettato nello spazio degli output desiderato di dimensione output_size
    L'output finale è una distribuzione di probabilità su tutti i possibili token di output per ciascun passo temporale della sequenza.
    '''
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        #permute(0,2,1) viene utilizzato per riordinare le dimensioni dell'output per adattarsi alla forma richiesta. 
        #In particolare, trasforma le dimensioni da (batch_size, seq_len, output_size) a (batch_size, output_size, seq_len)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    

'''
CLASSI DA ME IMPLEMENTATE
'''

'''
PUNTO 1
classe LSTM
'''

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        #permute(0,2,1) viene utilizzato per riordinare le dimensioni dell'output per adattarsi alla forma richiesta. 
        #In particolare, trasforma le dimensioni da (batch_size, seq_len, output_size) a (batch_size, output_size, seq_len)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    

'''
PUNTO 2/3
classe LSTM dropout. 
Sono stati aggiunti due dropout layer: uno dopo l'embedding layer e uno prima l'ultimo layer lineare.
Nell'__init__ chiamo la classe nn.Dropout passandogli prima il vettore di embedding e poi quello di output.
Nel forward viene calcolato l'embedding della sequenza di input e il risultato viene passato alla maschera del dropout,
cosa analoga per l'ultimo layer. 
'''

class LM_LSTM_dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_dropout, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        #aggiungi dropout layer 
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index

        #aggiungi dropout layer 
        self.out_dropout = nn.Dropout(out_dropout)

        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        #dropout
        emb_dropout = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb_dropout)
        #dropout
        output_dropout = self.out_dropout(lstm_out)
        output = self.output(output_dropout).permute(0,2,1)
        return output
    
