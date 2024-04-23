#write all the other required functions (taken from the notebook and/or written on your own) 
#needed to complete the exercise.

import torch
import torch.nn as nn
import math


'''
Implementa il ciclo di addestramento del modello,  
calcolando la perdita e aggiornando i pesi del modello utilizzando l'ottimizzatore su ogni iterazione. 
Alla fine restituisce la perdita media per token sull'intero set di dati di addestramento
'''
def train_loop(data, optimizer, criterion, model, clip=5):
    #imposta il modello in modalità di addestramento. 
    model.train()#Importante perché alcuni strati, come quelli che utilizzano il dropout o la normalizzazione batch, hanno comportamenti diversi durante l'addestramento e la valutazione.
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() #azzera i gradienti per evitare che i gradienti si accumulino da iterazioni precedenti
        output = model(sample['source']) #passa al modello sample['souce'] come input e trova output associato 
        loss = criterion(output, sample['target']) #calcola la perdita tra l'output previsto e il target
        loss_array.append(loss.item() * sample["number_tokens"]) #aggiunge la perdita normalizzata per il numero di token al loss_array 
        number_of_tokens.append(sample["number_tokens"]) #aggiunge il numero di token dell'esempio corrente alla lista number_of_tokens
        
        #COMPUTE STOCHASTIC GRADIENT
        loss.backward() # calcola i gradienti di tutti i parametri rispetto alla perdita
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #limita la magnitudine dei gradienti per evitare il problema della "vanishing" o "exploding" gradient
        
        optimizer.step() #aggiorna i pesi del modello usando l'ottimizzatore
        
        
    return sum(loss_array)/sum(number_of_tokens) #restituisce la perdita media per token sull'intero set di dati di addestramento



'''
calcola la perplexity e la perdita media per token del modello sui dati di valutazione, NON AGGIORNA I PESI DEL MODELLO
'''
def eval_loop(data, eval_criterion, model):
    model.eval() # imposta il modello in modalità di valutazione
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source']) #calcola output
            loss = eval_criterion(output, sample['target']) #calcola loss
            loss_array.append(loss.item()) #aggiunge perdita 
            number_of_tokens.append(sample["number_tokens"]) #aggiunge numero token 
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens)) #calcola perplexity
    loss_to_return = sum(loss_array) / sum(number_of_tokens) #calola perdita media per token 
    return ppl, loss_to_return #restituisce la perplexity e la perdita media per token


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


