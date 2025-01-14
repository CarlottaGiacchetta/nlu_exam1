#write all the other required functions (taken from the notebook and/or written on your own) 
#needed to complete the exercise.
import torch
import torch.nn as nn

from conll import evaluate
from sklearn.metrics import classification_report

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

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
    
        
        
        #sample['attention'] = torch.stack(sample['attention'])
        optimizer.zero_grad() # Zeroing the gradient
        intent, slots = model(sample['utterance'], sample['attention'])
   

        loss_intent = criterion_intents(intent, sample['intent'])
        
        loss_slot = criterion_slots(slots, sample['slots'])
        
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
        
            intents, slots = model(sample['utterance'], sample['attention'])
            loss_intent = criterion_intents(intents, sample['intent'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())#vettore delle loss
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()] #intent predictions
            gt_intents = [lang.id2intent[x] for x in sample['intent'].tolist()] #intent ground truth
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)


            for id_seq, seq in enumerate(output_slots):

                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist() #retrieves the original token IDs for the sequence up to its actual length, ensuring only relevant tokens are considered
                gt_ids = sample['slots'][id_seq].tolist() #etrieves the ground truth slot labels for the entire sequence, which includes padding
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]] #converts the ground truth slot IDs to their corresponding slot labels using a mapping (lang.id2slot), considering only the non-padded portion of the sequence
                '''utterance =  tokenizer.decode(utt_ids)
                utterance = utterance.split()'''

                utterance = [elem for elem in tokenizer.convert_ids_to_tokens(utt_ids)] #converts the list of token IDs (utt_ids) back to their string representation using the tokenizer
                
                to_decode = seq[:length].tolist() #prepares the predicted slot labels for evaluation by ensuring only the labels corresponding to the actual tokens (non-padded) are considered
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]) #appending Ground Truth and Predictions
                
                '''
                Similar to ref_slots, this loop constructs a list of tuples for the predicted slot labels (tmp_seq), 
                where each tuple consists of a token and its predicted slot label. This list is then appended to hyp_slots
                '''
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

                
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    #print('\n report intent')
    #print(report_intent)
    
    return results, report_intent, loss_array