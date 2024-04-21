import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader 
import torch.optim as optim
from utils import VariationalDropout


DEVICE = 'cpu:0' #cuda:0

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    
    '''
    In sintesi, questo modello accetta una sequenza di token come input, li trasforma in vettori di embedding, 
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
PRIMA PARTE 
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
    


'''
SECONDA PARTE 
'''
'''   
class LM_LSTM_wt(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_wt, self).__init__()
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

        #weight tying
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

class LM_LSTM_wt(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_wt, self).__init__()
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

        #weight tying
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
    


class LM_LSTM_wt_dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_wt_dropout, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        #aggiungi dropout layer 
        self.emb_dropout = VariationalDropout()

        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        #self.pad_token = pad_index

        #aggiungi dropout layer 
        self.out_dropout = VariationalDropout()

        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)

        #weight tying
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
