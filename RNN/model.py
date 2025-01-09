import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class RNN(nn.Module):
    
    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int, batchSize: int):
        super(RNN, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.batchSize = batchSize
        
        self.i2h = nn.Linear(inputSize, hiddenSize, bias=False)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, outputSize)
        
    def forward(self, x, h):
        x = self.i2h(x)
        h = self.h2h(h)
        h = torch.tanh(x + h)
        o = self.h2o(h)
        
        return o, h
    
    def initialize(self, b = 1):
        return torch.zeros(b, self.hiddenSize, requires_grad=False)
    
    
class LSTMCell(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        
        super(LSTMCell, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        # Input gate
        self.W_ii = nn.Parameter(torch.tensor(hiddenSize, inputSize))
        self.W_hi = nn.Parameter(torch.tensor(hiddenSize, hiddenSize))
        self.b_i = nn.Parameter(torch.tensor(hiddenSize))
        
        # Forget gate
        self.W_if = nn.Parameter(torch.tensor(hiddenSize, inputSize))
        self.W_hf = nn.Parameter(torch.tensor(hiddenSize, hiddenSize))
        self.b_f = nn.Parameter(torch.tensor(hiddenSize))
        
        # Cell gate
        self.W_ig = nn.Parameter(torch.tensor(hiddenSize, inputSize))
        self.W_hg = nn.Parameter(torch.tensor(hiddenSize, hiddenSize))
        self.b_g = nn.Parameter(torch.tensor(hiddenSize))
        
        # Output gate
        self.W_io = nn.Parameter(torch.tensor(hiddenSize, inputSize))
        self.W_ho = nn.Parameter(torch.tensor(hiddenSize, hiddenSize))
        self.b_o = nn.Parameter(torch.tensor(hiddenSize))
        
        self.init_weight()
        
    def init_weight(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
            
    def forward(self, x, h):
        hPrev, cPrev = h
        i_t = torch.sigmoid(x @ self.W_ii.T +  + hPrev @ self.W_hi.T + self.b_i)
        f_t = torch.sigmoid(x @ self.W_if.T +  + hPrev @ self.W_hf.T + self.b_f)
        c_t_tmp = torch.tanh(x @ self.W_ig.T +  + hPrev @ self.W_hg.T + self.b_g)
        o_t = torch.sigmoid(x @ self.W_io.T + hPrev @ self.W_ho.T + self.b_o)
        
        c_t = f_t * cPrev + i_t * c_t_tmp
        h_t = torch.tanh(c_t) * o_t
        
        return h_t, (h_t, c_t)
    
class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, layerNum):
        super(LSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.layerNum = layerNum
        
        self.cells = nn.ModuleList([LSTMCell(inputSize, hiddenSize) if i == 0 else LSTMCell(hiddenSize, hiddenSize) for i in range(layerNum)])
        self.fc = nn.Linear(hiddenSize, 1)
        
    def forward(self, x):
        B, seqLen, _ = x.shape
        h = [torch.zeros(B, self.hiddenSize).to(x.device) for _ in range(self.layerNum)]
        c = [torch.zeros(B, self.hiddenSize).to(x.device) for _ in range(self.layerNum)]
        
        for t in range(seqLen):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], (h[i], c[i]) = cell(x_t, (h[i], c[i]))
                x_t = h[i]
                
        o = self.fc(h[-1])
        return o
    
class GRUCell(nn.Module):
    def __init__(self, inputSize, hiddenSize, bias=True):
        super(GRUCell, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.bias = bias
        
        self.x2h = nn.Linear(inputSize, hiddenSize * 3, bias)
        self.h2h = nn.Linear(hiddenSize, hiddenSize * 3, bias)
        
        self.initialize()
        
    def initialize(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def forward(self, x, hidden):
        x = x.view(-1, x.shape[1])
        
        x = self.x2h(x).squeeze()
        h = self.h2h(hidden).squeeze()
        
        i_r, i_z, i_n = x.chunk(3, 1)
        h_r, h_z, h_n = h.chunk(3, 1)
        
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        h_tmp = F.tanh(i_n + h_n * r)
        o = h_tmp + z * (hidden - h_tmp)
        
        return o
        
class GRU(nn.Module):
    def __init__(self, inputSize, hiddenSize, layerNum):
        super(GRU, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.layerNum = layerNum
        
        self.cells = nn.ModuleList([GRUCell(inputSize, hiddenSize) if i == 0 else GRUCell(hiddenSize, hiddenSize) for i in range(layerNum)])
        self.fc = nn.Linear(hiddenSize, 1)
        
    def forward(self, x):
        B, seqLen, _ = x.shape
        h = [torch.zeros(B, self.hiddenSize).to(x.device) for _ in range(self.layerNum)]
        
        for t in range(seqLen):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]
                
        o = self.fc(h[-1])
        return o