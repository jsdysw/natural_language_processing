import torch

from params import MAX_LEN, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, EMBED_DIM, OUTPUT_SIZE, DROP_OUT_P, dataset_path, VOCAB_SIZE
from model import LSTM
from prepare_data import prepare_data
from vocab import Vocab
from utils import preprocess_string

def main():
    device = 'cpu'

    print('load vocab')
    vocab = Vocab(VOCAB_SIZE)
    vocab.load_dic_from_file('word2index.txt', 
                             'index2word.txt', 
                             'word2count.txt')

    print('define model')
    model = LSTM(HIDDEN_SIZE,
                 NUM_LAYERS,
                 EMBED_DIM,
                 VOCAB_SIZE,
                 OUTPUT_SIZE,
                 device,
                 DROP_OUT_P).to(device)


    review  = "I really liked this movie, I want to watch it again, Story was so good"
    
    print('review : ',  review) 
    
    print('load model ')
    model.load_state_dict(torch.load('./snapshot/imdb_txt_classification.pt'))
    
    print('inference ')
    result = test_review(model, review, vocab, device, MAX_LEN)
    print('result : ', result) 
    


def test_review(model, review, vocab, device, MAX_LEN):
    model.eval()
    threshold = 0.5
    
    x = preprocess_string(review, vocab, MAX_LEN)
    x.to(device)
    print('x shape ', x.shape)
    h, c = model.init_hidden(1)
    
    out, _ = model(x, h, c)        
    if out > threshold:
        return True
    else: 
        return False


if __name__ == '__main__':
    main()
