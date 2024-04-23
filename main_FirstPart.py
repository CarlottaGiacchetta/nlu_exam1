import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from utils import read_file, get_vocab, collate_fn, train_loop, eval_loop, init_weights
from nn_classes import LM_RNN, LM_LSTM, LM_LSTM_dropout
from language_classes import Lang, PennTreeBank
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

cuda = True
if cuda == True:
    DEVICE = 'cuda:0' 
else:
    DEVICE = 'cpu:0' 


train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

#compute the vocabulary 
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
#compute the len of the vocabulary and print
print('VOCABULARY LEN: \t', len(vocab))
#creo la classe che mi permetterà poi di chiamare la funzione get_vocab facendo lang.get_vocab(train_raw, ["<pad>", "<eos>"])
lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang) #è una classe che cotinee il dataset di train
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)


# Dataloader instantiation
# You can reduce the batch_size if the GPU memory is not enough
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

#SET PARAMETRI MODELLO
hid_size = 300
emb_size = 200
lrs = [1]
clip = 5 # Clip the gradient
n_epochs = 100
patience = 3 #è il numero di epoche di tolleranza dopo le quali si interrompe l'addestramento se non c'è miglioramento

vocab_len = len(lang.word2id)



punto_esericizio_lista = [2,3]

for lr in lrs:

    for punto_esericizio in punto_esericizio_lista:
        print(punto_esericizio)
        print(lr)
        if punto_esericizio == 0:
            model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
        elif punto_esericizio == 1:
            model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
        elif punto_esericizio == 2:
            model = LM_LSTM_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
        elif punto_esericizio == 3:
            model = LM_LSTM_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')



        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,n_epochs))
        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0: #per ogni epoca (visto che eil resto della divisione per 0 è 1)
                sampled_epochs.append(epoch) #aggiungi epoca alla lista
                losses_train.append(np.asarray(loss).mean()) #aggiunge la media delle perdite di addestramento di quell'epoca alla lista 
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model) #valutazione del modello 
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev) #aggiornamento della barra di avanzamento, serve per il print
                if  ppl_dev < best_ppl: #se la perplexity è migliorata
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        #Dopo la fine del ciclo di addestramento, il modello migliore viene trasferito alla CPU 
        best_model.to(DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
        print('Test ppl: ', final_ppl)


