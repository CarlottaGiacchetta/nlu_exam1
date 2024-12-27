# Mandatory Exam Exercise1 - LM
## Part 1 (4 points)
In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment. For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
One of the important tasks of training a neural network is hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular the learning rate). These are two links to the state-of-the-art papers which use vanilla RNN  [paper1](https://ieeexplore.ieee.org/document/5947611), [paper2](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf).
### Mandatory requirements:
For the following experiments, the perplexity must be below 250 (PPL < 250).

- **Replace RNN with a Long-Short Term Memory (LSTM) network** --> [link](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- **Add two dropout layers:** --> [link](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
  - One after the embedding layer,
  - One before the last linear layer
- **Replace SGD with AdamW** --> [link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)


# Part 2 (11 points)
### Mandatory requirements:
For the following experiments the perplexity must be below 250 (PPL < 250) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the LM_RNN in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:

- **Weight Tying**
- **Variational Dropout** (no DropConnect)
- **Non-monotonically Triggered AvSGD**
These techniques are described in this [paper](https://openreview.net/pdf?id=SyyGPP0TZ).


# Mandatory Exam Exercise2 - NLU

## Part 1 (4 points)

As for the LM project, you have to apply these two modifications incrementally. Also in this case, you may have to play with the hyperparameters and optimizers to improve the performance.

### Modify the baseline architecture Model IAS by:
- Adding bidirectionality
- Adding a dropout layer

### Evaluation Metrics:
- **Intent Classification:** Accuracy
- **Slot Filling:** F1 score with CoNLL

**Dataset to use:** ATIS

## Part 2 (11 points)

Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. You can refer to this paper to have a better understanding of how to implement this: [Multi-task Learning for Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909). In this, one of the challenges is to handle the sub-tokenization issue.

**Note:** The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.

### Models to experiment with:
- BERT-base
- BERT-large

### Evaluation Metrics:
- **Intent Classification:** Accuracy
- **Slot Filling:** F1 score with CoNLL

**Dataset to use:** ATIS
