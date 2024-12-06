import numpy as np
from collections import deque, defaultdict

class corpusData:
    
    def __init__(self, corpusPath, minFreq):
        self.corpusPath = corpusPath
        self.corpusFile = open(self.corpusPath)
        self.minFreq = minFreq
        self.id2freq = dict()
        self.wordNum = 0
        self.wordSum = 0
        self.sentenceNum = 0
        self.idx2word = defaultdict(int)
        self.word2idx = defaultdict(int)
        self.initialize()
        self.sample_table = []
        self.initializeNeg()
        self.pairs = deque()
        
    def initialize(self):
        wordFreq = dict()
        for line in self.corpusFile:
            line = line.strip().split(' ')
            self.wordSum += len(line)
            self.sentenceNum += 1
            
            for word in line:
                wordFreq[word] += 1
                
        idx = 0
        for word, freq in wordFreq.items():
            if freq < self.minFreq:
                self.wordSum -= freq
                continue
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            self.id2freq[idx] = freq
            idx += 1
        self.wordNum = len(self.word2idx)
        
    def initializeNeg(self):
        tableSize = 1e8
        negProb = np.array(list(self.id2freq.values())) ** 0.75
        probSum = sum(negProb)
        negProb = negProb / probSum
        
        wordCnt = np.round(negProb * tableSize)
        for idx, freq in enumerate(wordCnt):
            self.sample_table += [idx] * int(freq)
        self.sample_table = np.array(self.sample_table)
        
    def getBatchPairs(self, batchSize, windowSize):
        while len(self.pairs) < batchSize:
            for _ in range(10000):
                self.corpusFile = open(self.corpusPath, encoding="utf-8")
                sentence = self.corpusFile.readline()
                if sentence == None or sentence == "":
                    continue
                idxes = []
                for word in sentence.strip().split(' '):
                    if word in self.word2idx:
                        idx = self.word2idx[word]
                        idxes.append(idx)
                        
                for i, wordIdx in enumerate(idxes):
                    contextsIdx = []
                    for j, contextIdx in enumerate(idxes[max(i-windowSize, 0):i+windowSize+1]):
                        if i == j:
                            continue
                        elif max(0, i-windowSize+1) <= j <= min(len(idxes), i+windowSize-1):
                            contextsIdx.append(contextIdx)
                    
                    if len(contextsIdx) == 0:
                        continue
                    self.pairs.append(contextsIdx, wordIdx)
                    
        resPairs = []
        for _ in range(batchSize):
            resPairs.append(self.pairs.popleft())
        return resPairs
    
    def getNegSample(self, posPairs, negCount):
        return np.random.choice(self.sample_table, size=(len(posPairs), negCount)).tolist()
    
    def countPairs(self, windowSize):
        return self.wordSum * (2*windowSize-1) - (self.sentenceNum-1) * (1+windowSize) * windowSize