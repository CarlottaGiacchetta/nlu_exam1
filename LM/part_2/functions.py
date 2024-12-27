#write all the other required functions (taken from the notebook and/or written on your own) 
#needed to complete the exercise.

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


'''
FUNZIONI IMPLEMENTATE DA ME PER IL PUNTO 3
'''

'''
Ho introdotto una variabile di controllo: control che all'inizio dell'allenamento è settata 
come False e se le "non-triggering conditions" vengono rispettate verrà swithcata a True
In particolare:
- Aggiunge la metrica di performance attuale ai logs se l'iterazione corrente è un multiplo di L e T è disattivato.
- Attiva l'averaging dei parametri e modifica il valore di T se la metrica di performance si deteriora rispetto al miglior valore precedente dopo n iterazioni.
  Fa questo perchè in questo modo cambai il parametro t0 nell'AVSGD e permette di aggiornare le medie.
  Le medie nel AVSGD sono aggiornate moltiplicando per un fattore mu: 
  mu = max(1, k-t0) e t0 è settato ad un valore molto alto, quindi le medie non vengon mai aggiornate 
  --> io vado a settare t0 = T ovvero il numero dell'epoca in cui inizia il trigger 
  --> da ora in poi mu != 1
'''
def check(k , L, T, t, logs, ppl_dev, n, optimizer, control):
    if k % L == 0 and T == 0:
        logs.append(ppl_dev)
        t = t + 1
        
        if t > n and ppl_dev > min(logs[:k-n]): #forse da cambiare con k-n-1
            T = k
            print('SWITCH CON AVERAGING')
            optimizer.param_groups[0]['t0'] = T 
            print(optimizer.param_groups[0]['t0'])
            control = True
    k = k + 1
    return k, T, t, logs, control 

'''
Implementa un ciclo di training che utilizza NT-AvSGD (Non-Triggered Average Stochastic Gradient Descent)
come strategia di ottimizzazione. Questo ciclo di addestramento utilizza il metodo ASGD (Average Stochastic
Gradient Descent) di PyTorch. In partciolare ASGD fa l'update dei pesi usando SGD ma ad ogni itezione,
se il parametro t0 è minore di k, calcola gli averaging dei parametri salvandoli in ['ax']. 
Quindi la mia idea è stata quella di salvarmi gli averagign ['ax'] e passarli al modello come parametri da ottimizzare.

nello specifico:
- Il modello viene impostato in modalità training.
- In ogni iterazione, calcola la perdita, esegue il backpropagation, e aggiorna i parametri.
- Se `control` è True, applica l'averaging dei parametri memorizzati nel stato dell'optimizer.
  Questi parametri di averaging saranno usati dall'iterazione successiva per aggiornare i parametri attuali del modello.
- La funzione restituisce la perdita media per token su tutti i dati di addestramento.

'''
def train_loop_NTAvSGD(data, optimizer, criterion, model, control, k, clip=5):
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
        
        for item in optimizer.state.items():
            item[1]['step'] = torch.tensor(k, dtype=torch.float32)
        
        optimizer.step() #aggiornamento parametri 

        #CASO IN CUI CONTROL E' TRUE --> USARE AVERAGING
        if control:  
            tmp = []
            '''
            lista temporanea che mi salva i valori averaging 
            '''
            for item in optimizer.state.items():
                ax = item[1]['ax']
                tmp.append(ax)
            '''
            sostituzione dei parametri con ax 
            '''
            i = 0
            for param in optimizer.param_groups:
                for p in param['params']:
                    p.data.copy_(tmp[i])
                    i = i + 1              
                           
    return sum(loss_array)/sum(number_of_tokens) #restituisce la perdita media per token sull'intero set di dati di addestramento



