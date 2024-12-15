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
import argparse

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
        
        
class cbowDataset(Dataset):
    
    def __init__(self, corpusData: corpusProcess, maxWindowSize, negSampleNum):
        super(cbowDataset, self).__init__()
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
        
        contexts = torch.IntTensor([self.corpusWordList[curIdx] for curIdx in range(startIdx, endIdx) if curIdx != wordIdx])
        try:
            negWordList = self.curThreading.negWordList
        except:
            negWordList = self.negProb.clone()
            self.curThreading.negWordList = negWordList
            
        negWordNum = self.negSampleNum * len(contexts)
        negExclude = [self.corpusWordList[curIdx] for curIdx in range(startIdx, endIdx)]
        negWordList[negExclude] = 0
        
        negWords = torch.multinomial(negWordList, negWordNum, replacement=True)
        negWordList[negExclude] = self.negProb[negExclude]
        
        return contexts, word, negWords
    
    
class cbowModel(nn.Module):
    def __init__(self, vocSize, embedSize):
        super(cbowModel, self).__init__()
        self.vocSize = vocSize
        self.embedSize = embedSize
        self.embedding = nn.Embedding(vocSize, embedSize)
        self.embeddingContext = nn.Embedding(vocSize, embedSize)
        
    def forward(self, posContext, word, negContext):
        inputEmb = self.embedding(word)
        inputEmb = torch.transpose(inputEmb, dim0=2, dim1=1)
        posEmb = self.embeddingContext(posContext)
        negEmb = self.embeddingContext(negContext)
        
        posProb = torch.squeeze(torch.bmm(posEmb, inputEmb), dim=2)
        posProb = torch.sum(nn.functional.logsigmoid(posProb), dim=1)
        
        negEmb = torch.neg(negEmb)
        negProb = torch.squeeze(torch.bmm(negEmb, inputEmb), dim=2)
        negProb = torch.sum(nn.functional.logsigmoid(negProb), dim=1)
        
        loss = posProb + negProb
        loss = torch.mean(torch.neg(loss))
        return loss
    
class word2vec_CBOW:
    
    def __init__(self, corpusData: corpusProcess, embSize):
        super(word2vec_CBOW, self).__init__()
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
            
        dataset = cbowDataset(self.corpusData, maxWindowSize, negSampleNum)
        dataloader = DataLoader(dataset, batchSize, shuffle=True)
        batchNum = len(dataloader)
        vocSize = len(self.corpusData.vocList)
        
        model = cbowModel(vocSize, self.embSize)
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
        model = cbowModel(vocSize, self.embSize)
        
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
    
def parse_args():
    parse = argparse.ArgumentParser(description='Select mode to run the word2vec')
    parse.add_argument('--mode', type=str, default="train", help='Operation mode')
    parse.add_argument('--corpusPath', type=str, default="data/text8.txt", help='Path of corpus')
    parse.add_argument('--ckptsPath', type=str, default="ckpts/word2vec_CBOW.pth", help='Path of checkpoints')
    parse.add_argument('--w_size', type=int, default=5, help='Size of context window')
    parse.add_argument('--neg_num', type=int, default=15, help='Number of negative sampling')
    parse.add_argument('--max_voc_size', type=int, default=10000, help='Maximum size of vocabulary')
    parse.add_argument('--emb_dim', type=int, default=100, help='Dimension of word embedding')
    parse.add_argument('--epochs', type=int, default=1, help='Number of epoch for training')
    parse.add_argument('--bs', type=int, default=32, help='Batch Size')
    parse.add_argument('--lr', type=int, default=0.2, help='Learning Rate')
    args = parse.parse_args()
    return args     
            
if __name__ == '__main__':
    args = parse_args()
    runMode = args.mode
    testList = ['two', 'america', 'computer', 'queen', 'king', 'woman', 'man', 'black', 'green', 'java']
    
    if runMode == 'train':
        with open(args.corpusPath, 'r', encoding='utf-8') as f:
            file = f.read()
            
        token = tokenizer()
        corpusData = corpusProcess(token, args.w_size)
        corpusData.loadData(file)
        
        word2vec = word2vec_CBOW(corpusData, args.emb_dim)
        word2vec.train(args.ckptsPath, args.w_size, args.neg_num, args.epochs, args.bs, args.lr)
        
    elif runMode == 'test':
        with open(args.corpusPath, 'r', encoding='utf-8') as f:
            file = f.read()
            
        token = tokenizer()
        corpusData = corpusProcess(token, args.max_voc_size)
        corpusData.loadData(file)
        
        word2vec = word2vec_CBOW(corpusData, args.emb_dim)
        word2vec.test(args.ckptsPath, testList)