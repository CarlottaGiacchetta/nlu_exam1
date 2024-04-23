import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from utils import read_file, get_vocab, collate_fn, train_loop_NTAvSGD, eval_loop, init_weights, train_loop
from nn_classes import  LM_LSTM_wt, LM_LSTM_wt_dropout
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
#sprint('VOCABULARY LEN: \t', len(vocab))
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
hid_size = 200
emb_size = 200
lrs = [1, 2, 10] 
clip = 5 # Clip the gradient
n_epochs = 100
patience = 3 #è il numero di epoche di tolleranza dopo le quali si interrompe l'addestramento se non c'è miglioramento

vocab_len = len(lang.word2id)


lista_perplexity = []

punto_esericizio_lista = [1, 2, 3]
punto_esericizio_lista = [3]


for lr in lrs:

    for punto_esericizio in punto_esericizio_lista:
        print(punto_esericizio)
        print(lr)
        if punto_esericizio == 1:
            model = LM_LSTM_wt(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.SGD(model.parameters(), lr=lr)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1) # così mi cambia la lr in modo automatico
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        elif punto_esericizio == 2:
            model = LM_LSTM_wt_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.SGD(model.parameters(), lr=lr)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1) # così mi cambia la lr in modo automatico
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        elif punto_esericizio == 3:
            model = LM_LSTM_wt_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
            model.apply(init_weights)
            #SET OTTIMIZZATORE
            optimizer = optim.ASGD(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        if punto_esericizio == 1 or punto_esericizio == 2:
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


        elif punto_esericizio == 3:


            losses_train = []
            losses_dev = []
            sampled_epochs = []
            best_ppl = math.inf
            best_model = None
            pbar = tqdm(range(1,n_epochs))
            #If the PPL is too high try to change the learning rate

            L = 2 #logging interval
            T = 0 #trigger interval
            k = 0 #step
            n = 5 #non-monotone interval
            t = 0
            control = False #variabile che fa lo switch tra SGD e AvSGD
            logs = []
            ppl_dev = None


            for epoch in pbar:
                #print('\nSTAMPO k', k)
                #print('STAMPO T', T)
                #print('STAMPO t', t)
                #print('STAMPO n', n)
                print('STAMPO control',control)
                #print('STAMPO logs', logs)
                loss = train_loop_NTAvSGD(train_loader, optimizer, criterion_train, model, control, clip)    
                
                if epoch % 1 == 0: #per ogni epoca (visto che eil resto della divisione per 0 è 1)
                    sampled_epochs.append(epoch) #aggiungi epoca alla lista
                    losses_train.append(np.asarray(loss).mean()) #aggiunge la media delle perdite di addestramento di quell'epoca alla lista 
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model) #valutazione del modello 
                   
                    #CHECK
                    if k % L == 0 and T == 0:
                        logs.append(ppl_dev)
                        t = t + 1
                        
                        if t > n and ppl_dev > min(logs[:k-n]): #forse da cambiare con k-n-1
                            T = k
                            print('SWITCH CON AVERAGING')
                            optimizer.param_groups[0]['t0'] = T 
                            for item in optimizer.state.items():
                                item[1]['step'] = k.to(torch.float32)
                                
                            print(optimizer.param_groups[0]['t0'])
                            control = True
                    
                    k = k + 1


                    losses_dev.append(np.asarray(loss_dev).mean())
                    pbar.set_description("PPL: %f" % ppl_dev) #aggiornamento della barra di avanzamento, serve per il print
                    if  ppl_dev < best_ppl: #se la perplexity è migliorata
                        best_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = 3
                    else:
                        patience -= 1
                    
                    #STOPPING CRITERIO
                    if patience <= 0: # Early stopping with patience
                        break # Not nice but it keeps the code clean

            #Dopo la fine del ciclo di addestramento, il modello migliore viene trasferito alla CPU 
            best_model.to(DEVICE)
            final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)   
            lista_perplexity.append({'perplexity': final_ppl, 'model': punto_esericizio, 'lr': lr}) 
            print('Test ppl: ', final_ppl)

    with open('results', 'w') as f:
        for i in lista_perplexity:
            f.write(f"MODEL: {i['model']}\n")
            f.write(f"PPL: {i['perplexity']}\n")
            f.write(f"lr: {i['lr']} \n\n")

