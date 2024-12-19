import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

torch.manual_seed(1)
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import argparse

class corpusData:
    
    def __init__(self, args):
        super(corpusData, self).__init__()
        self.corpusPath = args.corpusPath
        text_file = open(args.corpusPath, 'r')
        raw_text = text_file.read().lower()
        text_file.close()
        self.token_text = word_tokenize(raw_text)
        self.len_token_text = len(self.token_text)
        
        # print("The number of tokens: ", self.len_token_text, '\n', self.token_text[:10])
        
        self.vocab = set(self.token_text)
        self.vocab_size = len(self.vocab)
        print("size of vocabulary: ", self.vocab_size)

        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.ix_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.contextSize = args.w_size
        self.co_occ_mat = np.zeros((self.vocab_size, self.vocab_size))
        
        for i in range(self.len_token_text):
            for dist in range(1, self.contextSize + 1):
                ix = self.word_to_ix[self.token_text[i]]
                if i - dist > 0:
                    left_ix = self.word_to_ix[self.token_text[i - dist]]
                    self.co_occ_mat[ix, left_ix] += 1.0 / dist
                if i + dist < self.len_token_text:
                    right_ix = self.word_to_ix[self.token_text[i + dist]]
                    self.co_occ_mat[ix, right_ix] += 1.0 / dist
           
        self.co_occs = np.transpose(np.nonzero(self.co_occ_mat))

        # print("shape of co-occurrence matrix:", self.co_occ_mat.shape)
        # print("non-zero co-occurrences:\n", self.co_occs)
        
class GloveModel(nn.Module):
    def __init__(self, vocabSize, comat, embSize, xmax, alpha, bs):
        super(GloveModel, self).__init__()
        self.vocabSize = vocabSize
        self.comat = comat
        self.embSize = embSize
        self.xmax = xmax
        self.alpha = alpha
        self.bs = bs
        
        self.embedding_word = nn.Embedding(vocabSize, embSize)
        self.embedding_context = nn.Embedding(vocabSize, embSize)
        
        self.biases_word = nn.Embedding(vocabSize, 1)
        self.biases_context = nn.Embedding(vocabSize, 1)
        
        for params in self.parameters():
            nn.init.uniform_(params, a = -0.5, b = 0.5)
            
    def forward(self, word, context):
        wordEmb = self.embedding_word(word)
        contextEmb = self.embedding_context(context)
        
        wordBias = self.biases_word(word).squeeze(1)
        contextBias = self.biases_context(context).squeeze(1)

        co_occurrences = torch.tensor([self.comat[word[i].item(), context[i].item()] for i in range(self.bs)])
        weights = torch.tensor([self.weight_fn(var) for var in co_occurrences])
        loss = torch.sum(torch.pow((torch.sum(wordEmb*contextEmb, dim=1)+wordBias+contextBias)-torch.log(co_occurrences), 2) * weights)
        return loss
        
    def weight_fn(self, x):
        if x < self.xmax:
            return (x / self.xmax) ** self.alpha
        return 1
    
    def embeddings(self):
        return self.embedding_word.weight.data + self.embedding_context.weight.data


def gen_batch(co_occs, bs=32):
    sample = np.random.choice(np.arange(len(co_occs)), size=bs, replace=False)
    v_vecs_ix, u_vecs_ix = [], []
    
    for chosen in sample:
        ind = tuple(co_occs[chosen])     
        lookup_ix_v = ind[0]
        lookup_ix_u = ind[1]
        
        v_vecs_ix.append(lookup_ix_v)
        u_vecs_ix.append(lookup_ix_u) 
        
    return torch.tensor(v_vecs_ix), torch.tensor(u_vecs_ix)


def train(comat, corpus: corpusData, args):
    losses = []
    model = GloveModel(corpus.vocab_size, comat, embSize=args.emb_dim, x_max=args.xmax, alpha=args.alpha)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = int(corpus.len_token_text/args.bs)
        print("Beginning epoch %d" %epoch)
        for batch in tqdm(range(num_batches)):
            model.zero_grad()
            data = gen_batch(model, args.bs)
            loss = model(*data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print('Epoch : %d, mean loss : %.02f' % (epoch, np.mean(losses)))
    state_dict = model.state_dict()
    torch.save(state_dict, args.ckptsPath)
    return model, losses

def get_word(word, model: GloveModel, word_to_ix):
    return model.embeddings()[word_to_ix[word]]

def closest(vec, word_to_ix, n=10):
    all_dists = [(w, torch.dist(vec, get_word(w, model, word_to_ix))) for w in word_to_ix]
    return sorted(all_dists, key=lambda t: t[1])[:n]

def parse_args():
    parse = argparse.ArgumentParser(description='Select mode to run the word2vec')
    parse.add_argument('--mode', type=str, default="train", help='Operation mode')
    parse.add_argument('--corpusPath', type=str, default="data/text8.txt", help='Path of corpus')
    parse.add_argument('--ckptsPath', type=str, default="ckpts/word2vec_CBOW.pth", help='Path of checkpoints')
    parse.add_argument('--w_size', type=int, default=10, help='Size of context window')
    parse.add_argument('--emb_dim', type=int, default=50, help='Dimension of word embedding')
    parse.add_argument('--epochs', type=int, default=50, help='Number of epoch for training')
    parse.add_argument('--bs', type=int, default=32, help='Batch Size')
    parse.add_argument('--lr', type=float, default=0.05, help='Learning Rate')
    parse.add_argument('--xmax', type=int, default=100, help='Maximum value for x')
    parse.add_argument('--alpha', type=float, default=0.75, help='Alpha')
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    corpus = corpusData(args)
    
    if args.mode == "train":
        model, losses = train(corpus.co_occ_mat, args)
    elif args.mode == "test":
        model = GloveModel(corpus.vocab_size, corpus.co_occ_mat, embSize=args.emb_dim, x_max=args.xmax, alpha=args.alpha)
        model.load_state_dict(torch.load(args.ckptsPath))
        model.eval()
        vector = get_word("lupov", model, corpus.word_to_ix)
        print(vector)
        
        closest(vector, corpus.word_to_ix)