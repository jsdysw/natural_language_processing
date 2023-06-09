import json

class Vocab:
    def __init__(self, vocab_size):
        self.word2index = {"PAD":0, "OOV":1}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"OOV"}
        self.n_words = 2  # Count PAD and OOV
        self.vocab_size = vocab_size

    def load_dic_from_file(self, w2i, i2w, w2c):
        with open(w2i, "r") as fp:
            self.word2index = json.load(fp)
        with open(w2c, "r") as fp:
            self.word2count = json.load(fp)
        with open(i2w, "r") as fp:
            self.index2word = json.load(fp)

        self.n_words = len(self.word2index)


    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def finishAddingWords(self):
        self.word2count = dict(sorted(self.word2count.items(), key=lambda item: item[1], reverse=True))
        idx = 2
        for key in self.word2count:
            if len(self.word2index) == self.vocab_size:
                break
            self.word2index[key] = idx
            self.index2word[idx] = key
            idx += 1

    def save2file(self,):
        with open("word2index.txt", "w") as fp:
            json.dump(self.word2index, fp) 
        with open("word2count.txt", "w") as fp:
            json.dump(self.word2count, fp) 
        with open("index2word.txt", "w") as fp:
            json.dump(self.index2word, fp) 

        
