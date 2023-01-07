import sys
sys.path.insert(3, './processing')
import finetune
import data_management
import BERT_tokenizer
import numpy as np
import torch
import torch.nn.functional as F

from transformers import BertForTokenClassification

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
        
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from operator import itemgetter

def BA_Model(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = BERT_tokenizer.tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(finetune.align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    probabilities = F.softmax(logits_clean, dim=-1).tolist()
    prediction_label = [data_management.ids_to_labels[i] for i in predictions]

    sentence_word = []
    id_rank = []
    count = 0
    label_count = 0
	
    x = 0
    input = str.split(sentence)
    #for i in input:
      #print(input[x] + ": " + prediction_label[x])
      #sentence_word.append(input[x])
      #for n in data_management.labels_to_ids:
          #if predictions[x] == data_management.labels_to_ids[n]:
              #for j in sentence_word:
                  #if j == input[x]:
                      #print(count)
                      #id_rank.append([sentence_word[x], prediction_label[x], probabilities[count][data_management.labels_to_ids[n]]])
                      #count = count + 1
      #x = x + 1
    #print(prediction_label)
    print(data_management.labels_to_ids)
    for i in input:
        id_rank.append([input[x], predictions[x], probabilities[x][predictions[x]]])
        x = x + 1
    for i in probabilities:
        print(i)
    print(sorted(id_rank, key=itemgetter(1, 2), reverse=True))
    #print(predictions)
    #print(input)
            
