# This code is from https://github.com/LogIntelligence/LogADEmpirical/blob/dev/logadempirical/models/lstm.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional

class DeepLog(nn.Module):
    # def __init__(self, hidden_size, num_layers, num_keys, padding_idx):
    #     super(DeepLog, self).__init__()
    #     self.hidden_size = hidden_size
    #     self.num_layers = num_layers
    #     self.embedding = nn.Embedding(num_keys + 2, hidden_size, padding_idx = padding_idx)
    #     self.lstm = nn.LSTM(hidden_size,
    #                         hidden_size,
    #                         num_layers,
    #                         batch_first=True)
    #     self.fc = nn.Linear(hidden_size, num_keys)

    # def forward(self, x):
    #     x = self.embedding(x)
    #     # h0 = torch.zeros(self.num_layers, x.size(0),
    #     #                  self.hidden_size).to(x.device)
    #     # c0 = torch.zeros(self.num_layers, x.size(0),
    #     #                  self.hidden_size).to(x.device)
    #     out, _ = self.lstm(x)
    #     out = self.fc(out[:, -1, :])
    #     return out
    
    def __init__(self, hidden_size, num_layers, num_keys):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(1,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        x = x.unsqueeze(-1).float()
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

        

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    


# log key add embedding
class LogAnomaly(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 vocab_size: int = 100,
                 embedding_dim: int = 300,
                 dropout: float = 0.5,
                 use_semantic: bool = True):
        super(LogAnomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_semantic = use_semantic
        self.embedding = None
        if not self.use_semantic:
            self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim)
            torch.nn.init.uniform_(self.embedding.weight)
            self.embedding.weight.requires_grad = True

        self.lstm0 = nn.LSTM(input_size=self.embedding_dim,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=False)

        self.lstm1 = nn.LSTM(input_size= self.vocab_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=False)
        self.fc = nn.Linear(2 * hidden_size, self.vocab_size)

    # def __init__(self, hidden_size: int = 128, num_layers: int = 2, vocab_size: int = 100, dropout: float = 0.5):
    #     super(LogAnomaly, self).__init__()
    #     self.hidden_size = hidden_size
    #     self.num_layers = num_layers
    #     self.vocab_size = vocab_size
    #     self.lstm0 = nn.LSTM(input_size=1,
    #                          hidden_size=hidden_size,
    #                          num_layers=num_layers,
    #                          batch_first=True,
    #                          dropout=dropout,
    #                          bidirectional=False)
    #     self.lstm1 = nn.LSTM(input_size=self.vocab_size,
    #                             hidden_size=hidden_size,
    #                             num_layers=num_layers,
    #                             batch_first=True,
    #                             dropout=dropout,
    #                             bidirectional=False)
    #     self.fc = nn.Linear(2 * hidden_size, self.vocab_size)

    def forward(self, x_seq, x_quant):
        # x_seq = x_seq.unsqueeze(-1).float()
        out0, _ = self.lstm0(x_seq)
        out1, _ = self.lstm1(x_quant)

        multi_out = torch.cat((out0[:, -1, :], out1[:, :]), -1)
        logits = self.fc(multi_out)
        # probabilities = torch.softmax(logits, dim=-1)
        # loss = None
        # if y is not None and self.criterion is not None:
        #     loss = self.criterion(logits, y.view(-1).to(device))

        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, batch, device="cpu"):
        del batch['label']
        return self.forward(batch, device=device).probabilities

    def predict_class(self, batch, top_k=1, device="cpu"):
        del batch['label']
        return torch.topk(self.forward(batch, device=device).probabilities, k=top_k, dim=1).indices