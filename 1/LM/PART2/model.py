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
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res




'''
CLASSI DA ME IMPLEMENTATE
'''


'''
PUNTO 1 
classe LSTM weight tying. Ho passato i pesi dell'embedding layer all'output layer.
Per fare questo c'è bisogno che l'embedding layer (emb_size) e l'outut layer (output_size) hanno la stessa dimensione
'''
 
class LM_LSTM_wt(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_wt, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

        #weight tying
        self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)

        #permute(0,2,1) viene utilizzato per riordinare le dimensioni dell'output per adattarsi alla forma richiesta. 
        #In particolare, trasforma le dimensioni da (batch_size, seq_len, output_size) a (batch_size, output_size, seq_len)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    

'''
PUNTO 2/3
classe LSTM weight tying e variational dropout. 
Sono stati aggiunti due dropout layer: uno dopo l'embedding layer e uno prima l'ultimo layer lineare.
Nell'__init__ chiamo la classe VariationalDropout da me implementata e che può essere trovata sotto.
Nel forward viene calcolato l'embedding della sequenza di imput e il risultato viene passato alla maschera del dropout,
cosa analoga per l'ultimo layer. 
'''

class LM_LSTM_wt_dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_wt_dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        #aggiungi dropout layer chimanado la funzione VariationalDropout
        self.emb_dropout = VariationalDropout()

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    


        #aggiungi dropout layer 
        self.out_dropout = VariationalDropout()

        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        #dropout
        emb_dropout = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb_dropout)
        #dropout
        output_dropout = self.out_dropout(lstm_out)
        output = self.output(output_dropout).permute(0,2,1)
        return output


'''
classe che  implementa il variational dropout; viene applicato lo stesso pattern di dropout
'''

class VariationalDropout(nn.Module): 
    def _init_(self): 
        super()._init_() 
 
    def forward(self, x, dropout=0.1): 
        if not self.training or not dropout: 
            return x 
        
        '''
        m è un tensore della stessa dimensione del secondo e terzo asse di x,
        che si distribuisce come una Bernoulli con probabilità 1 - dropout. 
        '''
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)

        '''
        creo la maschera da m 
        ''' 
        mask = Variable(m, requires_grad=False) / (1 - dropout) 
        mask = mask.expand_as(x) 

        '''
        applico la mascher all'input --> spegne in modo automatico alcuni neuroni
        '''
        return mask * x 