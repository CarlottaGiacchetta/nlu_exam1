#put all the functions needed to preprocess and load the dataset
import json
import torch
from sklearn.model_selection import train_test_split
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


'''
The split_DevSet function separates the training data into a training set and a development (dev) set 
using stratified sampling based on intent labels. 
It initially ensures that intents occurring only once are retained in the training set. 
Then, it performs stratified sampling to divide the remaining training data into 90% training and 10% dev sets. 
The resulting train_raw and dev_raw lists contain the separated data, while the function also returns 
the intents (labels) for both the dev and training sets, as well as for the given test set.
'''

def split_DevSet(tmp_train_raw, test_raw):
    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)
    labels = []
    inputs = []
    mini_train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify --> stratified sampling 
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    return train_raw, dev_raw, y_test, y_dev, y_train


'''
Create a dictionary that maps the words and labels in the training set to unique  integers $\geq$ 0, called indexes.
That is:
- One dictionary for mapping words to ids (w2id)
- One dictionary for mapping slot labels to ids (slot2id)
- One dictionary for mapping intent labels to ids (intent2id)

With w2id map the sentence in `sent` into the computed indexes.
'''

def dictionries(train_raw, dev_raw, test_raw):
    w2id = {'pad':PAD_TOKEN, 'unk': 1}
    slot2id = {'pad':PAD_TOKEN}
    intent2id = {}
    # Map the words only from the train set
    # Map slot and intent labels of train, dev and test set. 'unk' is not needed.
    for example in train_raw:
        for w in example['utterance'].split():
            if w not in w2id:
                w2id[w] = len(w2id)  #chiave: parola, valore: lunghezza attuale del dizionario 
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)
            
    for example in dev_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)
            
    for example in test_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    return w2id, slot2id, intent2id


'''
The padding function ensures all sequences have the same length by first finding the maximum 
length among them and then appending the special 'PAD' token to shorter sequences until 
they match this length.
'''
def padding(sequences):

    splitted = [seq.split() for seq in sequences]
    max_len = max([len(seq) for seq in splitted])
    padded_seq = []
    for seq in splitted:
        diff = max_len - len(seq)
        if diff != 0:
            # Depending on the side of the addition we can get left or right padding
            padded_seq.append(seq + ['PAD'] * diff)
        else:
            padded_seq.append(seq)
    return padded_seq



def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item




