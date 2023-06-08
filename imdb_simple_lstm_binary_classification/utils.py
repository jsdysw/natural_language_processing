import torch

from params import device, MAX_LEN


def encode_dataset(vocab, X_train, X_valid, X_test):
    X_train_encode = [indexesFromSentence(vocab, row['review']) for _, row in X_train.iterrows()]
    X_valid_encode = [indexesFromSentence(vocab, row['review']) for _, row in X_valid.iterrows()]
    X_test_encode = [indexesFromSentence(vocab, row['review']) for _, row in X_test.iterrows()]

    X_train_encode_pad = [[0]*(MAX_LEN-len(x)) + x if len(x)<MAX_LEN\
                     else x[0:MAX_LEN] for x in X_train_encode]

    X_valid_encode_pad = [[0]*(MAX_LEN-len(x)) + x if len(x)<MAX_LEN\
                        else x[0:MAX_LEN] for x in X_valid_encode]

    X_test_encode_pad = [[0]*(MAX_LEN-len(x)) + x if len(x)<MAX_LEN\
                        else x[0:MAX_LEN] for x in X_test_encode]

    return X_train_encode_pad, X_valid_encode_pad, X_test_encode_pad

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] if word in vocab.word2index.keys() else vocab.word2index['OOV']\
            for word in sentence.split(' ')]

def tensorFromSentence(vocab, sentence, device, MAX_LEN = 300):
    indexes = indexesFromSentence(vocab, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def preprocess_string(review, vocab, device, MAX_LEN):    
    review = review.lower()
    # review = review.replace('<[^>]*>','', regex=True)
    # review = review.replace(r'[^a-zA-Z ]','', regex=True)
    # review = review.replace('^ +', '', regex=True) # white space -> empty value

    x = indexesFromSentence(vocab, review)
    x_encode_pad = [[0]*(MAX_LEN-len(x)) + x if len(x)<MAX_LEN else x[0:MAX_LEN]]

    return torch.tensor(x_encode_pad, dtype=torch.long, device=device).view(1, -1)