from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, embed_dim, n_vocab, output_size, device, dropout_p=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p, inplace=False)
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, cell):
        # print('1. x : ', x.shape)
        out = self.embed(x)
        # print('2. embed : ', out.shape)
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
            
        # print('3. lstm : ', out.shape)
        # print('   3-1. hiddem : ', hidden.shape)
        # print('   3-2. cell : ', cell.shape)

        self.dropout(out)
        out = out[:, -1, :]
        # print('reshape ', out.shape)
        out = self.fc(out)
        out = self.sigmoid(out)
        # print('4. fc : ', out.shape)
        return out, (hidden, cell)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden, cell