from transformers import BertTokenizer, BertModel
from pprint import pprint
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
import csv
import copy

from utils import load_data, split_DevSet, collate_fn, modify_slot
from model import Lang, IntentsAndSlots, BertForJointIntentAndSlot
from functions import train_loop, eval_loop


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0") # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0


'''
READ FILES
'''
tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
print('Train samples:', len(tmp_train_raw))
print('Test samples:', len(test_raw))

pprint(tmp_train_raw[0])

'''
CRATE A DEV SET 
'''

train_raw, dev_raw, y_test, y_dev, y_train = split_DevSet(tmp_train_raw, test_raw)
print('='*89)
print('INTENT DISTRIBUTIONS')
# Intent distributions
print('Train:')
pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
print('Dev:'), 
pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
print('Test:') 
pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
print('='*89)
# Dataset size
print('TRAIN size:', len(train_raw))
print('DEV size:', len(dev_raw))
print('TEST size:', len(test_raw))

'''
TOKENIZATION
'''
mode = 'paper'
#mode = None

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=False) # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased", force_download=False) # Download the model

if mode == 'paper':
    
    train_raw1 = train_raw
    dev_raw1 = dev_raw
    test_raw1 = test_raw

    count = 0 
    for raw in train_raw1:
        utt = raw['utterance'].split()
        slot = raw['slots'].split()

        if len(utt) == len(slot):
            pass
        else:
            count += 1   

else:
    #ho bisogno di riallineare gli slots
    train_raw1 = modify_slot(train_raw, tokenizer)
    dev_raw1 = modify_slot(dev_raw, tokenizer)
    test_raw1 = modify_slot(test_raw, tokenizer)


corpus = train_raw1 + dev_raw1 + test_raw1
intents = set([line['intent'] for line in corpus])
slots = set(sum([line['slots'].split() for line in corpus],[]))
lang = Lang(intents, slots,  cutoff=0)

train_dataset = IntentsAndSlots(train_raw1, tokenizer, lang, mode)
dev_dataset = IntentsAndSlots(dev_raw1, tokenizer, lang, mode)
test_dataset = IntentsAndSlots(test_raw1, tokenizer, lang, mode)
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)



'''
FINE TUNING
'''
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)




lr_list = [5e-4]
prob_drop_list = [0.5]
epochs_list = [50]




count = 0
for lr in lr_list:

    for prob_drop in prob_drop_list:

        best_model = None

        for epoch in epochs_list:
            

            #path_saveresults = os.path.join('NLU','PART2','RISULTATI', f"{count}.csv")
            count += 1

            modellooo = BertForJointIntentAndSlot(model, num_intents = len(lang.intent2id), num_slots = len(lang.slot2id), dropout_prob = prob_drop).to(device)
            optimizer = torch.optim.Adam(modellooo.parameters(), lr=lr)

            best_f1 = 0
            patience = 3
            losses_train = []
            losses_dev = []
            sampled_epochs = []
            for x in tqdm(range(1, epoch)):
                loss = train_loop(train_loader, optimizer, criterion_slots,
                                                    criterion_intents, modellooo, clip = 5)
                if x % 5 == 0: # We check the performance every 5 epochs
                    sampled_epochs.append(x)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                                criterion_intents, modellooo, lang, tokenizer)
                    losses_dev.append(np.asarray(loss_dev).mean())

                    f1 = results_dev['total']['f']
                    # For decreasing the patience you can also use the average between slot f1 and intent accuracy
                    if f1 > best_f1:
                        best_f1 = f1
                        # Here you should save the model
                        patience = 3
                    else:
                        patience -= 1
                    if patience <= 0: # Early stopping with patience
                        break # Not nice but it keeps the code clean

            best_model = copy.deepcopy(modellooo).to('cpu')

            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                                        criterion_intents, modellooo, lang, tokenizer)
            print('Slot F1: ', results_test['total']['f'])
            print('Intent Accuracy:', intent_test['accuracy'])
        
        #Dopo la fine del ciclo di addestramento, il modello migliore viene trasferito alla CPU 
        #best_model.to(device)
        #model_path = f'esperimento2.pt'
        #torch.save(model.state_dict(), model_path)
        


        #    #SALVA I RISULTATI
        #    with open(path_saveresults, mode='w', newline='') as file_csv:
        #        # Crea un writer CSV
        #        csv_writer = csv.writer(file_csv)
#
        #        # Scrivi l'intestazione (parametri)
        #        csv_writer.writerow(['Model', f'{modellooo}'])
        #        csv_writer.writerow(['lr', f'{lr}'])
        #        csv_writer.writerow(['dorp prob', f'{prob_drop}'])
        #        csv_writer.writerow(['epoch', f'{epoch}'])
        #        csv_writer.writerow(['Slot F1', f'{results_test["total"]["f"]}'])
        #        csv_writer.writerow(['Intent Accuracy', f'{intent_test["accuracy"]}'])
#
        #    plt.figure(num = epoch, figsize=(8, 5)).patch.set_facecolor('white')
        #    plt.title('Train and Dev Losses')
        #    plt.ylabel('Loss')
        #    plt.xlabel('Epochs')
        #    plt.plot(sampled_epochs, losses_train, label='Train loss')
        #    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
        #    plt.legend()
        #    plt.show()
        #    plt.savefig(os.path.join('NLU','PART2','RISULTATI', f"{count}.png"))
            
            

