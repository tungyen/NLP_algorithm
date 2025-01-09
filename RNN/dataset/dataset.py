import torch
from torch.utils.data import Dataset

class textDataset(Dataset):
    
    def __init__(self, textData: str, seqLength: int):
        self.chars = sorted(list(set(textData)))
        self.dataSize = len(textData)
        self.vocSize = len(self.chars)
        
        self.idx2char = {i:ch for i, ch in enumerate(self.chars)}
        self.char2idx = {ch:i for i, ch in enumerate(self.chars)}
        self.seqLength = seqLength
        self.X = self.string2vector(textData)
        
    @property
    def X_string(self):
        return self.vector2string(self.X)
        
    def string2vector(self, string: str):
        vector = list()
        for s in string:
            vector.append(self.char2idx[s])
        return vector
    
    def vector2string(self, vector: list[int]):
        string = ""
        for v in vector:
            string += self.idx2char[v]
        return string
    
    def __len__(self):
        return int(len(self.X) / self.seqLength -1)
    
    def __getitem__(self, index):
        s = index * self.seqLength
        e = (index + 1) * self.seqLength
        
        X = torch.tensor(self.X[s:e]).float()
        Y = torch.tensor(self.X[s+1:e+1]).float()
        
        return X, Y