#write the calls to the functions needed to output the results asked by the exercise
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
import csv

from utils import load_data, split_DevSet, dictionries, padding, collate_fn
from model import Lang, IntentsAndSlots,  ModelIAS, ModelIAS_Bidirectional, ModelIAS_Bidirectional_drop
from functions import init_weights, train_loop, eval_loop


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
CRATE DICTIONARIES FOR MAPPING
'''
print('='*89)
print('MAPPING AND PADDING')
w2id, slot2id, intent2id = dictionries(train_raw, dev_raw, test_raw)
sent = 'I wanna a flight from Toronto to Kuala Lumpur'
mapping = [w2id[w] if w in w2id else w2id['unk'] for w in sent.split()] #se la parola non c'è nel dizionario, mette 1
print(sent, mapping)
print('# Vocab:', len(w2id)-2) # we remove pad and unk from the count
print('# Slots:', len(slot2id)-1)
print('# Intent:', len(intent2id))


words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw # We do not wat unk labels,
                                        # however this depends on the research purpose
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])
lang = Lang(words, intents, slots, cutoff=0)

sequences = ['I saw a man with a telescope', 
            'book me a flight', 
            'I want to see the flights from Milan to Ibiza']
padded_seq = padding(sequences)
print('\n')
for i in range(len(sequences)):
    print(sequences[i], padded_seq[i])

'''
CRATE THE DATASETS
'''
train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)


'''
TRAINING
'''
hid_size_emb_size = [(200, 300), (300, 500)]
lr = 0.0001 # learning rate
clip = 5 # Clip the gradient
lr_list = [0.0001, 0.01, 1]
prob_drop_list = [0.1, 0.5]

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)


criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

multiple_run = True
model_list = [ModelIAS, ModelIAS_Bidirectional, ModelIAS_Bidirectional_drop]


count = 0




for modelloo in model_list:

    for lr in lr_list:
        
        for prob_drop in prob_drop_list:
            
            for hidemb in hid_size_emb_size:

                path_saveresults = os.path.join('NLU','PART1','RISULTATI', f"{count}")

                count = count + 1
                print(f"MODELLO: {modelloo}, \t learning rate: {lr}, \t probabilità dropout: {prob_drop}, \t hidden & embedding size: {hidemb}")

                hid_size = hidemb[0]
                emb_size = hidemb[1]

        

                model = modelloo(hid_size, out_slot, out_int, emb_size, vocab_len, prob_drop, pad_index=PAD_TOKEN).to(device)
                model.apply(init_weights)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                if multiple_run == False:
                    n_epochs = 200
                    patience = 3
                    losses_train = []
                    losses_dev = []
                    sampled_epochs = []
                    best_f1 = 0
                    for x in tqdm(range(1,n_epochs)):
                        loss = train_loop(train_loader, optimizer, criterion_slots,
                                        criterion_intents, model, clip=clip)
                        if x % 5 == 0: # We check the performance every 5 epochs
                            sampled_epochs.append(x)
                            losses_train.append(np.asarray(loss).mean())
                            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                                        criterion_intents, model, lang)
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

                    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                            criterion_intents, model, lang)
                    print('Slot F1: ', results_test['total']['f'])
                    print('Intent Accuracy:', intent_test['accuracy'])

                    


                else:
            
                    out_slot = len(lang.slot2id)
                    out_int = len(lang.intent2id)
                    vocab_len = len(lang.word2id)

                    n_epochs = 200
                    runs = 5

                    slot_f1s, intent_acc = [], []
                    for x in tqdm(range(0, runs)):
                        model = modelloo(hid_size, out_slot, out_int, emb_size,
                                        vocab_len, prob_drop, pad_index=PAD_TOKEN).to(device)
                        model.apply(init_weights)

                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
                        criterion_intents = nn.CrossEntropyLoss()

                        patience = 3
                        losses_train = []
                        losses_dev = []
                        sampled_epochs = []
                        best_f1 = 0
                        for x in range(1,n_epochs):
                            loss = train_loop(train_loader, optimizer, criterion_slots,
                                            criterion_intents, model)
                            if x % 5 == 0:
                                sampled_epochs.append(x)
                                losses_train.append(np.asarray(loss).mean())
                                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                                            criterion_intents, model, lang)
                                losses_dev.append(np.asarray(loss_dev).mean())
                                f1 = results_dev['total']['f']

                                if f1 > best_f1:
                                    best_f1 = f1
                                else:
                                    patience -= 1
                                if patience <= 0: # Early stopping with patient
                                    break # Not nice but it keeps the code clean

                        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                                criterion_intents, model, lang)
                        intent_acc.append(intent_test['accuracy'])
                        slot_f1s.append(results_test['total']['f'])
                    slot_f1s = np.asarray(slot_f1s)
                    intent_acc = np.asarray(intent_acc)
                    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
                    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
                #SALVA I RISULTATI
                with open(path_saveresults, mode='w', newline='') as file_csv:
                    # Crea un writer CSV
                    csv_writer = csv.writer(file_csv)

                    # Scrivi l'intestazione (parametri)
                    csv_writer.writerow(['Model', f'{modelloo}'])
                    csv_writer.writerow(['lr', f'{lr}'])
                    csv_writer.writerow(['dorp prob', f'{prob_drop}'])
                    csv_writer.writerow(['hid size', f'{hid_size}'])
                    csv_writer.writerow(['emb size', f'{emb_size}'])
                    csv_writer.writerow(['Slot F1', f'{results_test["total"]["f"]}'])
                    csv_writer.writerow(['Intent Accuracy', f'{intent_test["accuracy"]}'])



    '''
    FOR SAVING THE MODEL 

    PATH = os.path.join("bin", model_name)
    saving_object = {"epoch": x, 
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(), 
                    "w2id": w2id, 
                    "slot2id": slot2id, 
                    "intent2id": intent2id}
    
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()
    torch.save(saving_object, PATH)
    '''


plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
plt.title('Train and Dev Losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(sampled_epochs, losses_train, label='Train loss')
plt.plot(sampled_epochs, losses_dev, label='Dev loss')
plt.legend()
plt.show()

'''
TRANSFORMERS
'''

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model

inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
pprint(inputs)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(inputs["input_ids"][0])
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))