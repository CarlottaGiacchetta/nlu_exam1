#the class of the model defined in PyTorch.
from collections import Counter
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


PAD_TOKEN = 0

'CLASSI GIà IMPLEMENTATE PER IL LOAD DEI DATI'

class Lang():
    def __init__(self, intents, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    


    
  
    

'''
 provides a structure to handle datasets that involve mapping utterances to numerical representations 
 of intents and slots. 
 It uses the lang Class for mapping: the utterances are mapped to word IDs (utt_ids), 
 the slots to slot IDs (slot_ids), and the intents to intent IDs (intent_ids)
'''
class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, tokenizer, lang, mode = None):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.mde = mode

        for x in dataset:
 
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
        if mode == 'paper':
            self.utt_ids, self.utt_attention = self.mapping_seq_new1(self.utterances, self.tokenizer)
        else:
            self.utt_ids, self.utt_attention = self.mapping_seq_new(self.utterances, self.tokenizer)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
            

    def __len__(self):
        return len(self.utterances)
    
    '''
    Accepts an index idx and retrieves the corresponding mapped utterance, slots, and intent.
    Returns a dictionary containing the tensors for 'utterance' and 'slots', and the intent ID as 'intent'.
    '''
    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx]) #The mapped utterance and slots are converted to PyTorch tensors
        att = torch.Tensor(self.utt_attention[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'attention': att}

        print(sample)
        exit()


        return sample

    # Auxiliary methods

    '''
    Maps a list of labels to their corresponding numerical IDs using a given mapper (dictionary).
    Returns the mapped list, using the unknown (unk) token for any label not found in the mapper.
    '''
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    '''
    Converts a list of sequences (e.g., utterances) to lists of numerical IDs based on a given mapper.
    Splits each sequence into individual tokens, mapping each one to its ID, 
    or using the unk token if not found.
    '''
    def mapping_seq_new(self, data, tokenizer): # Map sequences to number

        inputs = []
        attention = []
        max = 0
        for seq in data:
            encoded = tokenizer(seq)
 
        for seq in data:
            encoded = tokenizer(seq) 
            attention_tmp = encoded['attention_mask']
            
            inputs_tmp = encoded['input_ids'] 

            attention.append(attention_tmp)
            inputs.append(inputs_tmp)


        
        return inputs, attention
    

    def mapping_seq_new1(self, data, tokenizer): # Map sequences to number
        inputs = []
        attention = []

        vocab = {}
        for seq in data:
            for word in seq.split():
                vocab[word]  = tokenizer.encode(word)[1]            

        for seq in data:
            input_tmp = [101]
            for word in seq.split():
                input_tmp.append(vocab[word])
            input_tmp.append(102)
            inputs.append(input_tmp)
            att = [1] * len(input_tmp)
            attention.append(att)

            if len(input_tmp) == len(att):
                pass
            else:
                print(input_tmp)
                print(len(input_tmp))
                print(tokenizer(seq)['attention_mask'])
                print(len(tokenizer(seq)['attention_mask']))



        return inputs, attention
    

    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    


class BertForJointIntentAndSlot(nn.Module):
    def __init__(self, model, num_intents, num_slots, dropout_prob):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout_prob)

        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict = True)


        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)


        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        slot_logits = slot_logits.permute(0, 2, 1)

        return intent_logits, slot_logits

