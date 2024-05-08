#the class of the model defined in PyTorch.
#the class of the model defined in PyTorch.
from collections import Counter
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


PAD_TOKEN = 0

'CLASSI GIà IMPLEMENTATE PER IL LOAD DEI DATI'

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        #
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        for elem in elements:
            tokenized_word = tokenizer.tokenize(elem)
            tokens = tokenizer.convert_tokens_to_ids(tokenized_word)
            vocab[elem] = tokens[0]
        print('VOCAB ITEMS')
        print(vocab.items())
        
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            tokenized_word = tokenizer.tokenize(elem)
            tokens = tokenizer.convert_tokens_to_ids(tokenized_word)
            vocab[elem] = tokens[0]
        print('OH CAZZ, è GIUSTA STA ROBA?????')
        print(vocab)
        return vocab
    

'''
 provides a structure to handle datasets that involve mapping utterances to numerical representations 
 of intents and slots. 
 It uses the lang Class for mapping: the utterances are mapped to word IDs (utt_ids), 
 the slots to slot IDs (slot_ids), and the intents to intent IDs (intent_ids)
'''
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
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
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
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



'''
defines a nn model for intent classification and slot fillng
'''
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, prob_drop, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)    
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent




class ModelIAS_Bidirectional(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, prob_drop, n_layer=1, pad_index=0):
        super(ModelIAS_Bidirectional, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index) #layer to convert words to vectors
        #metto bidirectional = Trues
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True) #LSTM encoder to process each utterance
        
        #ci sono due output: (per il bi-directional va duplicata la dimensione)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)#output per slot filling
        self.intent_out = nn.Linear(hid_size * 2, out_int)#output per intent classification

        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        #convert utterance in embedding
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)#exclude padding tokens
        # Process the batch --> passes the packed embeddings through the LSTM (encder), retrieving the encoded representation.
        packed_output, (hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence --> extracts the encoded sequence for slot filling
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        #Final state for both directions
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits 
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    




class ModelIAS_Bidirectional_drop(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, prob_drop, n_layer=1, pad_index=0):
        super(ModelIAS_Bidirectional_drop, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index) #layer to convert words to vectors
        #metto bidirectional = Trues
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True) #LSTM encoder to process each utterance
        
        #ci sono due output: (per il bi-directional va duplicata la dimensione)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)#output per slot filling
        self.intent_out = nn.Linear(hid_size * 2, out_int)#output per intent classification

        # Dropout layer How/Where do we apply it?
        self.emb_dropout = nn.Dropout(prob_drop)

        self.out_dropout = nn.Dropout(prob_drop)

    def forward(self, utterance, seq_lengths):
        #convert utterance in embedding
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        utt_emb = self.emb_dropout(utt_emb) # Apply dropout after embedding
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)#exclude padding tokens
        # Process the batch --> passes the packed embeddings through the LSTM (encder), retrieving the encoded representation.
        packed_output, (hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence --> extracts the encoded sequence for slot filling
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        #add the dropout layer 
        utt_encoded = self.out_dropout(utt_encoded)
    
        #Final state for both directions
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        last_hidden = self.out_dropout(last_hidden)

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits 
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    





