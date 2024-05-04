#the class of the model defined in PyTorch.
from collections import Counter
import torch
import torch.utils.data as data


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
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
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
