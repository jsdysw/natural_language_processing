import os

import torch
from torch import utils, nn
from tqdm import tqdm
import numpy as np

from prepare_data import prepare_data
from vocab import Vocab
from utils import encode_dataset, preprocess_string
from dataset import TextDataset
from model import LSTM

from params import device, dataset_path, MAX_LEN, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, EMBED_DIM, OUTPUT_SIZE, DROP_OUT_P, LR, EPOCHS 

def main():
    print('device : ', device)
    print('prepare dataset')
    X_train, y_train, X_test, y_test, X_valid, y_valid = prepare_data(dataset_path)

    print('build vocab')
    vocab = Vocab()
    for _, row in X_train.iterrows():
        vocab.addSentence(row['review'])

    print('enocde dataset')
    X_train_encode_pad, X_valid_encode_pad, X_test_encode_pad = encode_dataset(vocab, X_train, X_valid, X_test)

    print('build dataloader')
    train_dataloader = utils.data.DataLoader(TextDataset(X_train_encode_pad, y_train), 
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)

    test_dataloader = utils.data.DataLoader(TextDataset(X_test_encode_pad, y_test),
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)

    valid_dataloader = utils.data.DataLoader(TextDataset(X_valid_encode_pad, y_valid),
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)
    
    print('define model')
    model = LSTM(HIDDEN_SIZE,
                 NUM_LAYERS,
                 EMBED_DIM,
                 vocab.n_words,
                 OUTPUT_SIZE,
                 device,
                 DROP_OUT_P,).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_bce = nn.BCELoss()

    print('train start')
    best_val_loss = None
    for e in range(1, EPOCHS+1):
        train(model, optimizer, train_dataloader, loss_bce)
        val_loss, val_accuracy = evaluate(model, valid_dataloader, loss_bce)

        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

        # save least loss model
        if not best_val_loss or val_loss < best_val_loss:
            print('model save')
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), './snapshot/imdb_txt_classification.pt')
            best_val_loss = val_loss
            

    print('test start')
    model.load_state_dict(torch.load('./snapshot/imdb_txt_classification.pt'))
    test_loss, test_acc = evaluate(model, test_dataloader, loss_bce)
    print('test loss: %5.2f | test acc: %5.2f' % (test_loss, test_acc))


def train(model, optimizer, train_dataloader, loss_bce):
    model.train()
    h, c = model.init_hidden(BATCH_SIZE)
    for batch in tqdm(train_dataloader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        out, _ = model(x, h, c)
        loss = loss_bce(out, y)
        loss.backward()
        optimizer.step()


def evaluate(model, valid_dataloader, loss_bce):
    model.eval()
    corrects, total_loss, total_n_iter = 0, 0, 0
    threshold = 0.5
    h, c = model.init_hidden(BATCH_SIZE)
    
    for batch in tqdm(valid_dataloader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        out, _ = model(x, h, c)        
        loss = loss_bce(out, y)        
        total_loss += loss.item()
        prediction_label = np.array(np.add(out.cpu().data.numpy(), threshold), dtype=np.int64)
        bCorrect = (prediction_label == y.cpu().data.numpy()).sum()        
        corrects += bCorrect
        total_n_iter += 1

    avg_loss = total_loss / total_n_iter
    avg_accuracy = corrects / (total_n_iter*BATCH_SIZE)
    return avg_loss, avg_accuracy
        

if __name__ == '__main__':
    main()