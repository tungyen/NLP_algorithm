import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from corpusProcess import corpusData

WINDOW_SIZE = 4
BATCH_SIZE = 64
MIN_COUNT = 3
EMB_DIM  = 100
LR = 0.02
NEG_COUNT = 4

class CBOW(nn.Module):
    def __init__(self, embSize, embDim):
        super(CBOW, self).__init__()
        self.embSize = embSize
        self.embDim = embDim
        self.embedding_context = nn.Embedding(self.embSize, self.embDim, sparse=True)
        self.embedding_word = nn.Embedding(self.embSize, self.embDim, sparse=True)
        self.initEmbedding()
        
    def initEmbedding(self):
        r = 0.5 / self.embDim
        self.embedding_context.weight.data.uniform_(-r, r)
        self.embedding_word.weight.data.uniform_(0, 0)
        
    def forward(self, pos_context, pos_word, neg_word):
        pos_context_emb = []
        for c in pos_context:
            c_emb = self.embedding_context(torch.LongTensor(c))
            c_embs = np.sum(c_emb.data.numpy(), axis=0).tolist()
            pos_context_emb.append(c_embs)
        pos_context_emb = torch.FloatTensor(pos_context_emb)
        pos_word_emb = self.embedding_word(torch.LongTensor(pos_word))
        neg_word_emb = self.embedding_word(torch.LongTensor(neg_word))
        
        score = F.logsigmoid(torch.sum(torch.mul(pos_context_emb, pos_word_emb).squeeze(), dim=1))
        negScore = F.logsigmoid((-1) * torch.sum(torch.bmm(neg_word_emb, pos_context_emb.unsqueeze(2)).squeeze(), dim=1))
        
        loss = torch.sum(score) + torch.sum(negScore)
        return -1 * loss
    
    def saveEmbedding(self, idx2word, outputPath):
        embedding = self.embedding_context.weight.data.numpy()
        file = open(outputPath, 'w')
        file.write('%d %d\n' % (self.embSize, self.embDim))
        for idx, word in idx2word.items():
            e = embedding[idx]
            e = ' '.join(map(lambda x: str(x), e))
            file.write('%s %s\n' % (word, e))
            
            
class word2vec_CBOW:
    def __init__(self, corpusPath, resPath):
        self.resPath = resPath
        self.data = corpusData(corpusPath, MIN_COUNT)
        self.model = CBOW(self.data.wordNum, EMB_DIM)
        self.lr = LR
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr)
        
    def train(self):
        print("Now start training the CBOW")
        pairCounts = self.data.countPairs(WINDOW_SIZE)
        print("Pair counts: ", pairCounts)
        batchs = pairCounts / BATCH_SIZE
        
        process = tqdm(range(int(batchs)))
        
        for i in process:
            posPairs = self.data.getBatchPairs(BATCH_SIZE, WINDOW_SIZE)
            pos_context = [pair[0] for pair in posPairs]
            pos_word = [pair[1] for pair in posPairs]
            neg_word = self.data.getNegSample(posPairs, NEG_COUNT)
            
            self.optim.zero_grad()
            loss = self.model(pos_context, pos_word, neg_word)
            loss.backward()
            self.optim.step()
            
            if i * BATCH_SIZE % 10000 == 0:
                self.lr = self.lr * (1.0 - 1.0 * i / batchs)
                for param in self.optim.param_groups:
                    param['lr'] = self.lr
                    
        self.model.saveEmbedding(self.data.idx2word, self.resPath)
        
        
if __name__ == '__main__':
    w2v = word2vec_CBOW(input_file_name='./data.txt', output_file_name="word_embedding.txt")
    w2v.train()
    
    f = open('word_embedding.txt')
    f.readline()
    all_embeddings = []
    all_words = []
    word2id = defaultdict()
    for i, line in enumerate(f):
        line = line.strip().split(' ')
        word = line[0]
        embedding = [float(x) for x in line[1:]]
        all_embeddings.append(embedding)
        all_words.append(word)
        word2id[word] = i
    all_embeddings = np.array(all_embeddings)
    
    while True:
        context = input("Context: ")
        contextsIds = []
        
        words = context.split(' ')
        check = True
        for word in words:
            if word not in word2id:
                print("Cannot find words")
                check = False
                break
            else:
                idx = word2id[word]
                contextsIds.append(idx)
        if not check:
            continue
        
        contexts_emb = []
        for context_id in contextsIds:
            context_emb = all_embeddings[context_id:context_id+1]
            contexts_emb.append(context_emb)
        contexts_emb_sum = np.sum(contexts_emb, axis=0)
        d = cosine_similarity(contexts_emb_sum, all_embeddings)[0]
        d = zip(all_words, d)
        d = sorted(d, key=lambda x: x[1], reverse=True)
        for w in d[:3]:
            print(w)