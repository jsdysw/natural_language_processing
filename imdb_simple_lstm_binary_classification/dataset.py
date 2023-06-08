from torch import utils
import torch

class TextDataset(utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.y = y
        self.x = x
        self.len = len(self.x)
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.x[idx]).to(torch.int64)
        y = torch.Tensor([self.y[idx]]).to(torch.float32)
        sample = {"x": x, "y": y}
        return sample
