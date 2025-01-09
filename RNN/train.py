import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

from model import *
from dataset.dataset import *

ckpts_path = "ckpts"
device = "cuda" if torch.cuda.is_available() else "cpu"
def train(model: RNN, textDataloader: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module):
    model.to(device)
    train_losses = {}
    model.train()
    
    print("start training!")
    
    for epoch in range(epochs):
        curLoss = list()
        for x, y in textDataloader:
            if x.shape[0] != model.batchSize:
                continue
            
            h = model.initialize(model.batchSize)
            x, y, h = x.to(device), y.to(device), h.to(device)
            model.zero_grad()
            loss = 0
            
            for c in range(x.shape[1]):
                o, h = model(x[:, c].reshape(x.shape[0], 1), h)
                loss += loss_fn(o, y[:, c].long())
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            curLoss.append(loss.detach().item() / x.shape[1])
            
        train_losses[epoch] = torch.tensor(curLoss).mean()
        print(f'=> epoch: {epoch + 1}, loss: {train_losses[epoch]}')
        if epoch % 100 == 0 and epoch != 0:
            weightPath = os.path.join(ckpts_path, 'epoch_{}.pth'.format(epoch+1))
            torch.save(model.state_dict(), weightPath)
        
if __name__ == '__main__':
    data = open('dataset/luen_yu_clean.txt', 'r').read()
    data = data.lower()
    
    seqLength = 25
    batch_size = 64
    hidden_size = 256
    
    text_dataset = textDataset(data, seqLength)
    text_dataloader = DataLoader(text_dataset, batch_size)
    rnnModel = RNN(1, hidden_size, len(text_dataset.chars), batch_size) # 1 because we enter a single number/letter per step.

    epochs = 1000
    loss = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(rnnModel.parameters(), lr = 0.001)
    train(rnnModel, text_dataloader, epochs, optimizer, loss)
            
            