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


DEVICE = 'cuda:0' #cpu:0

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



class VariationalDropout(nn.Module): 
    def _init_(self): 
        super()._init_() 
 
    def forward(self, x, dropout=0.1): 
        if not self.training or not dropout: 
            return x 
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout) 
        mask = Variable(m, requires_grad=False) / (1 - dropout) 
        mask = mask.expand_as(x) 
        return mask * x
    
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
        
        #prima iterazione chiami SGD
        if control:
            #print('CONTROL == TRUE YEEE')
            tmp = []
      
            

            for item in optimizer.state.items():
                ax = item[1]['ax']
                tmp.append(ax)

            i = 0
            for param in optimizer.param_groups:
                for p in param['params']:
                    p.data.copy_(tmp[i])
                    i = i + 1

        optimizer.step()
        
        
        
    return sum(loss_array)/sum(number_of_tokens), control#restituisce la perdita media per token sull'intero set di dati di addestramento


'''


def train_loop_NTAvSGD(data, optimizer, criterion, model, control, clip=5):
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
        
        if control:  
            #print('CONTROL == TRUE YEEE')
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

        optimizer.step()
        
        
        
    return sum(loss_array)/sum(number_of_tokens) #restituisce la perdita media per token sull'intero set di dati di addestramento



