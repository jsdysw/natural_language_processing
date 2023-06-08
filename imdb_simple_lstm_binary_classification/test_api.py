import torch

from params import MAX_LEN, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, EMBED_DIM, OUTPUT_SIZE, DROP_OUT_P, dataset_path
from model import LSTM
from prepare_data import prepare_data
from vocab import Vocab
from utils import preprocess_string

def main():
    print('prepare dataset')
    X_train, _, _, _, _, _ = prepare_data(dataset_path)

    print('build vocab')
    vocab = Vocab()
    for _, row in X_train.iterrows():
        vocab.addSentence(row['review'])


    print('define model')
    model = LSTM(HIDDEN_SIZE,
                 NUM_LAYERS,
                 EMBED_DIM,
                 180358,
                 OUTPUT_SIZE,
                 'cpu',
                 DROP_OUT_P).to('cpu')


    review  = "I hate this movie"
    
    print('review : ',  review) 
    
    print('load model ')
    model.load_state_dict(torch.load('./snapshot/imdb_txt_classification.pt'))
    
    print('inference ')
    result = test_review(model, review, vocab, MAX_LEN)
    print('result : ', result) 
    


def test_review(model, review, vocab, MAX_LEN):
    model.eval()
    threshold = 0.5
    
    x = preprocess_string(review, vocab, 'cpu', MAX_LEN)
    print('x shape ', x.shape)
    h, c = model.init_hidden(1)
    
    out, _ = model(x, h, c)        
    if out > threshold:
        return True
    else: 
        return False


if __name__ == '__main__':
    main()