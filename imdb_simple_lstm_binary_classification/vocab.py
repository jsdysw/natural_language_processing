class Vocab:
    def __init__(self):
        self.word2index = {"PAD":0, "OOV":1}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"OOV"}
        self.n_words = 2  # Count PAD, ST and OOV

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1