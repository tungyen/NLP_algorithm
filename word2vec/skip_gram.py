import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy
import random
import threading
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
import os

class tokenizer:
    def __init__(self):
        super(tokenizer, self).__init__()
        
    def splitWords(self, sentence: str):
        sentence = sentence.strip()
        sentence = sentence.lower()
        wordList = sentence.split(' ')
        
        return wordList
    
class corpusProcess:
    def __init__(self, tokenizer: tokenizer, maxVocabSize):
        super(corpusProcess, self).__init__()
        
        self.tokenizer = tokenizer
        self.maxVoc = maxVocabSize
        
        self.oriWords = list()
        self.encodedWords = list()
        
        self.word2idx = dict()
        self.idx2word = dict()
        
        self.vocList = list()
        self.vocCount = list()
        
    def loadData(self, sentence):
        oriWords = self.tokenizer.splitWords(sentence)
        
        word2count = Counter(oriWords)
        word2count = dict(word2count.most_common(self.maxVoc-1))
        word2count['<UNK>'] = len(oriWords) - sum(word2count.values())
        
        vocList = sorted(word2count, key=word2count.get, reverse=True)
        vocCount = list()
        word2idx = dict()
        idx2word = dict()
        
        for idx, word in enumerate(vocList):
            vocCount.append(word2count[word])
            word2idx[word] = idx
            idx2word[idx] = word
            
        unknownIdx = word2idx['<UNK>']
        encodeWords = [word2idx.get(word, unknownIdx) for word in oriWords]
        
        self.oriWords = oriWords
        self.encodedWords = encodeWords
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocCount = vocCount
        self.vocList = vocList
        
        
class skipGramDataset(Dataset):
    
    def __init__(self, corpusData: corpusProcess, maxWindowSize, negSampleNum):
        super(skipGramDataset, self).__init__()
        corpusWordList = corpusData.encodedWords
        corpusWordNum = len(corpusWordList)
        
        datasetNum = corpusWordNum - (2 * maxWindowSize)
        centerOffset = maxWindowSize
        
        vocCount = corpusData.vocCount
        vocCount_ = np.array(vocCount, dtype=np.float32)
        negProb = vocCount_ ** 0.75
        negProb = negProb / np.sum(negProb)
        
        self.datasetNum = datasetNum
        self.centerOffset = centerOffset
        self.maxWindowSize = maxWindowSize
        self.corpusWordList = corpusWordList
        self.corpusWordListTensor = torch.FloatTensor(corpusWordList)
        self.corpusWordNum = corpusWordNum
        
        self.negSampleNum = negSampleNum
        self.negProb = torch.FloatTensor(negProb)
        self.curThreading = threading.local()
        
    def __len__(self):
        return self.datasetNum
    
    def __getitem__(self, index):
        wordIdx = index + self.centerOffset
        word = torch.IntTensor([self.corpusWordList[wordIdx]])
        
        startIdx = wordIdx - self.maxWindowSize
        endIdx = wordIdx + self.maxWindowSize + 1
        
        posWordList = torch.IntTensor([self.corpusWordList[curIdx] for curIdx in range(startIdx, endIdx) if curIdx != wordIdx])
        try:
            negWordList = self.curThreading.negWordList
        except:
            negWordList = self.negProb.clone()
            self.curThreading.negWordList = negWordList
            
        negWordNum = self.negSampleNum * len(posWordList)
        negExclude = [self.corpusWordList[curIdx] for curIdx in range(startIdx, endIdx)]
        negWordList[negExclude] = 0
        
        negWords = torch.multinomial(negWordList, negWordNum, replacement=True)
        negWordList[negExclude] = self.negProb[negExclude]
        
        return word, posWordList, negWords
    
    
class skipGramModel(nn.Module):
    def __init__(self, vocSize, embedSize):
        super(skipGramModel, self).__init__()
        self.vocSize = vocSize
        self.embedSize = embedSize
        self.embedding = nn.Embedding(vocSize, embedSize)
        self.embeddingContext = nn.Embedding(vocSize, embedSize)
        
    def forward(self, inputWord, pos, neg):
        inputEmb = self.embedding(inputWord)
        inputEmb = torch.transpose(inputEmb, dim0=2, dim1=1)
        posEmb = self.embeddingContext(pos)
        negEmb = self.embeddingContext(neg)
        
        posProb = torch.squeeze(torch.bmm(posEmb, inputEmb), dim=2)
        posProb = torch.sum(nn.functional.logsigmoid(posProb), dim=1)
        
        negEmb = torch.neg(negEmb)
        negProb = torch.squeeze(torch.bmm(negEmb, inputEmb), dim=2)
        negProb = torch.sum(nn.functional.logsigmoid(negProb), dim=1)
        
        loss = posProb + negProb
        loss = torch.mean(torch.neg(loss))
        return loss
    
class word2vec_skipgram:
    
    def __init__(self, corpusData: corpusProcess, embSize):
        super(word2vec_skipgram, self).__init__()
        self.corpusData = corpusData
        self.embSize = embSize
        
    def train(self, resPath: str, maxWindowSize, negSampleNum, epochNum, batchSize, lr=0.1):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        dataset = skipGramDataset(self.corpusData, maxWindowSize, negSampleNum)
        dataloader = DataLoader(dataset, batchSize, shuffle=True)
        batchNum = len(dataloader)
        vocSize = len(self.corpusData.vocList)
        
        model = skipGramModel(vocSize, self.embSize)
        model = model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr)
        model.train()
        
        for epoch in tqdm(range(epochNum)):
            for batchIdx, (inputWord, pos, neg) in enumerate(dataloader):
                inputWord = inputWord.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                
                loss = model(inputWord, pos, neg)
                loss.backward()
                opt.step()
                opt.zero_grad()
                
                if batchIdx % 500 == 0:
                    print(f'loss: {loss.item():>10f}  [{batchIdx+1:>5d}/{batchNum:>5d}]')
                    
        state_dict = model.state_dict()
        torch.save(state_dict, resPath)
        
    def test(self, ckptsPath, testWords, nearestNum = 10):
        vocSize = len(self.corpusData.vocList)
        model = skipGramModel(vocSize, self.embSize)
        
        if not os.path.exists(ckptsPath):
            print("You should train the model first!")
            return
        
        model.load_state_dict(torch.load(ckptsPath))
        model.eval()
        
        word2idx = self.corpusData.word2idx
        idx2word = self.corpusData.idx2word
        
        weights = model.embedding.weight.detach().numpy()
        
        for word in testWords:
            if word not in word2idx:
                print(f'{word}: []')
                continue
            
            idx = word2idx[word]
            w = weights[idx]
            
            similarity = np.array([scipy.spatial.distance.cosine(curW, w) for curW in weights])
            idxes = similarity.argsort()
            idxes = idxes[1:nearestNum+1]
            nearest = [idx2word[i] for i in idxes]
            print(f'{word}: {nearest}')
            
            
if __name__ == '__main__':
    corpusPath = './text8.txt'
    ckptsPath = './word2vec_skipgram.pth'
    maxWindowSize = 5
    negSampleNum = 15
    maxVocSize = 10000
    embSize = 100
    epoch = 1
    batchSize = 32
    lr = 0.2
    
    runMode = 'test'
    # runMode = 'train'
    testList = ['two', 'america', 'computer', 'queen', 'king', 'woman', 'man', 'black', 'green', 'java']
    
    if runMode == 'train':
        with open(corpusPath, 'r', encoding='utf-8') as f:
            file = f.read()
            
        token = tokenizer()
        corpusData = corpusProcess(token, maxVocSize)
        corpusData.loadData(file)
        
        word2vec = word2vec_skipgram(corpusData, embSize)
        word2vec.train(ckptsPath, maxWindowSize, negSampleNum, epoch, batchSize, lr)
        
    elif runMode == 'test':
        with open(corpusPath, 'r', encoding='utf-8') as f:
            file = f.read()
            
        token = tokenizer()
        corpusData = corpusProcess(token, maxVocSize)
        corpusData.loadData(file)
        
        word2vec = word2vec_skipgram(corpusData, embSize)
        word2vec.test(ckptsPath, testList)