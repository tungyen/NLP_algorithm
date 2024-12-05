from collections import defaultdict, Counter

class BPE:
    
    def __init__(self, corpus: list[str], vocab_size: int, maxIter: int = 1000):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = []
        self.wordCnt = Counter()
        self.splits = {}
        self.merges = {}
        self.maxIter = maxIter
        
        
    def train(self):
        for doc in self.corpus:
            words = doc.split()
            self.wordCnt += Counter(words)
            
        for word in self.wordCnt:
            self.splits[word] = list(word) + ["</w>"]
            
        vocab = set()
        for word in self.wordCnt:
            vocab |= set(list(word))
        vocab.add("</w>")
        
        self.vocab = list(vocab)
        self.vocab.sort()
        
        cnt = 0
        while len(self.vocab) < self.vocab_size and cnt < self.maxIter:
            pairs = self.getPairs()
            
            if len(pairs) == 0:
                print("There is no pairs available")
                break
            
            pair = max(pairs, key=pairs.get)
            self.update(pair[0], pair[1])
            self.merges[pair] = pair[0] + pair[1]
            self.vocab.append(pair[0] + pair[1])
            cnt += 1
            
    def getPairs(self):
        pairs = defaultdict(int)
        for word, freq in self.wordCnt.items():
            split = self.splits[word]
            for i in range(len(split)-1):
                pairs[(split[i], split[i+1])] += freq
                
        return pairs
    
    def update(self, l: str, r: str):
        for word, word_split in self.splits.items():
            
            update_split = []
            idx = 0
            while idx < len(word_split):
                if word_split[idx] == l and idx < len(word_split)-1 and word_split[idx+1] == r:
                    update_split.append(l+r)
                    idx += 2
                else:
                    update_split.append(word_split[idx])
                    idx += 1
            self.splits[word] = update_split
            
    def tokenize(self, s: str) -> list[str]:
        splits = [list(t) + ["</w>"] for t in s.split()]
        
        for l, r in self.merges:
            for i, split in enumerate(splits):
                update_split = []
                idx = 0
                while idx < len(split):
                    if split[idx] == l and idx < len(split)-1 and split[idx+1] == r:
                        update_split.append(l+r)
                        idx += 2
                    else:
                        update_split.append(split[idx])
                        idx += 1
                        
                splits[i] = update_split
                
        return sum(splits, [])
    
    
if __name__ == '__main__':
    corpus = ["highest", "higher", "lower", "lowest", "cooler", "coolest"]
    bpe = BPE(corpus, vocab_size=17)
    bpe.train()
    print(bpe.tokenize(" ". join(corpus)))