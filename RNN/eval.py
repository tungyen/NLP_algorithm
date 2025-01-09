import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

from model import *
from dataset.dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def textGeneration(model: RNN, dataset: textDataset, pred_length: int = 100):
    model.eval()
    model.to(device)
    vec = [random.randint(0, len(dataset.chars) -1)]
    pred = dataset.vector2string(vec)
    h = model.initialize(model.batchSize)
    
    for _ in range(pred_length - 1):
        c = torch.Tensor([dataset.char2idx[pred[-1]]])
        
        x, h = c.to(device), h.to(device)
        o, h = model(x, h)
        
        o = nn.functional.softmax(o, 1)
        print(o.shape)
        result = torch.multinomial(o, 1).item()
        pred += dataset.idx2char[result]

    return pred

if __name__ == '__main__':
    
    data = open('dataset/luen_yu_clean.txt', 'r').read()
    data = data.lower()
    
    seqLength = 25
    batch_size = 1
    hidden_size = 256
    
    text_dataset = textDataset(data, seqLength)
    text_dataloader = DataLoader(text_dataset, batch_size)
    rnnModel = RNN(1, hidden_size, len(text_dataset.chars), batch_size)
    weightPath = os.path.join("ckpts", "epoch_901.pth")
    rnnModel.load_state_dict(torch.load(weightPath, map_location=device))
    
    print(textGeneration(rnnModel, text_dataloader.dataset))
    