#write the calls to the functions needed to output the results asked by the exercise
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
#importo funzioni/classi presenti negli altri file
from functions import  train_loop, train_loop_NTAvSGD, eval_loop, init_weights, check
from utils import read_file, get_vocab, collate_fn
from model import Lang, PennTreeBank, LM_LSTM_wt, LM_LSTM_wt_dropout

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")


'''
READ FILE, COMPUTE VOCABULARY
'''
train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")


vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
lang = Lang(train_raw, ["<pad>", "<eos>"])
train_dataset = PennTreeBank(train_raw, lang) #è una classe che cotinee il dataset di train
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

'''
SET PARAMETRI DEL MODELLO 
'''
#PARAMETRI CHE POSSO CAMBIARE 
hid_size = 600
emb_size = 600
lr = 15
clip = 5 # Clip the gradient
n_epochs = 100
patience = 6 #è il numero di epoche di tolleranza dopo le quali si interrompe l'addestramento se non c'è miglioramento

#VARIABILI UTILI PER IL PUNTO 3
L = 2 #logging interval
T = 0 #trigger interval
k = 0 #step
n = 5 #non-monotone interval
t = 0
control = False #variabile che fa lo switch tra SGD e AvSGD
logs = []
ppl_dev = None

vocab_len = len(lang.word2id)

#VARIABILI CONTROLLO CICLO 
punto_esericizio_lista = [1, 2, 3]



for punto_esericizio in punto_esericizio_lista:
    patience = 6
    print('ESRCIZIO NUMERO: \t',punto_esericizio)
    print('------------------------PARAMETRI DEL MODELLO------------------------')
    print('--> learning rate: \t',lr)
    print('--> (hid_size, emb_size): \t',(hid_size, emb_size))
    print('--> clip: \t',clip)
    print('--> patience: \t',patience)
    if punto_esericizio == 3:
        print('--> logging interval: \t',L)
        print('--> trigger interval: \t',T)
        print('--> non-monotone interval: \t',n)


    '''
    PUNTO 1 : CHIAMO CLASSE DI LSTM CON SOLO WEIGHT TYING E SGD COME OTTIMIZZATORE
    PUNTO 2 : CHIAMO CLASSE DI LSTM CON  WEIGHT TYING E VARIATIONAL DROPOUT E SGD COME OTTIMIZZATORE
    PUNTO 3 : CHIAMO CLASSE DI LSTM CON  WEIGHT TYING E VARIATIONAL DROPOUT E NT-AvSGD COME OTTIMIZZATORE
    '''
    if punto_esericizio == 1:
        model = LM_LSTM_wt(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)
        #SET OTTIMIZZATORE
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    

    elif punto_esericizio == 2:
        model = LM_LSTM_wt_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)
        #SET OTTIMIZZATORE
        optimizer = optim.SGD(model.parameters(), lr=lr)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1) # così mi
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    
    elif punto_esericizio == 3:
        model = LM_LSTM_wt_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)
        #SET OTTIMIZZATORE
        optimizer = optim.ASGD(model.parameters(), lr=lr)
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

        
        '''
        CHIAMA UN TRAIN LOOP DIVERSO SULLA BASE DELLL'OTTIMZZATORE CHE SI VUOLE USARE
        '''
        if punto_esericizio == 1 or punto_esericizio == 2: #SGD
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)  
        elif punto_esericizio == 3: #NT-AvSGD
            loss = train_loop_NTAvSGD(train_loader, optimizer, criterion_train, model, control, k, clip)

        
        
        if epoch % 1 == 0: #per ogni epoca (visto che eil resto della divisione per 0 è 1)
            sampled_epochs.append(epoch) #aggiungi epoca alla lista
            losses_train.append(np.asarray(loss).mean()) #aggiunge la media delle perdite di addestrament
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model) #valutazione del modello 

            '''
            FA IL CHECK PER IL TRIGGER -> se rispetta le condizioni cambai control in True 
            '''
            if punto_esericizio == 3:
                print('CONTROL --> ', control)
                k, T, t, logs, control  = check(k , L, T, t, logs, ppl_dev, n, optimizer, control)
                print('PARAMETRO STEP: \t',list(optimizer.state.values())[0]['step'])
                print('PARAMETRO MU: \t',list(optimizer.state.values())[0]['mu'])



            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev) #aggiornamento della barra di avanzamento, serve pe
            if  ppl_dev < best_ppl: #se la perplexity è migliorata
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 6
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    #Dopo la fine del ciclo di addestramento, il modello migliore viene trasferito alla CPU 
    best_model.to(DEVICE)
    print(os.listdir('.'))    
    model_path = f'esperimento{punto_esericizio}.pt'
    torch.save(model.state_dict(), model_path)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    
    
    
    