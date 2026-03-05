import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, lengths):
        # lstm_out: [B, N_max, H]
        scores = self.attn(lstm_out).squeeze(-1)             # [B, N_max]
        mask = torch.arange(lstm_out.size(1))[None, :].to(lengths.device) >= lengths[:, None]
        scores.masked_fill_(mask, -float('inf'))             # ignore padding
        weights = torch.softmax(scores, dim=1)               # [B, N_max]
        night_emb = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # [B, H]
        return night_emb, weights

class LSTM_architecture(nn.Module):
    def __init__(self, num_classes):
        super(LSTM_architecture,self).__init__()
        
        self.lstm = nn.LSTM(input_size=289, hidden_size=289, batch_first=True) #289
        self.attn_pool = AttentionPooling(hidden_dim=289)
        self.fc1 = nn.Linear(289, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(128,num_classes)

        self.num_classes = num_classes
        self.m = nn.Sigmoid()
        #self.m = nn.Softmax(dim=1)
      
    def forward(self, x, lengths):
        packed_out, (h, _) = self.lstm(x)  # h: [1, B, 128]
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        
        h, attn_weights = self.attn_pool(out, lengths)
        #print(h.shape)
        
        #print(h.shape)
        x = h.squeeze(0)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc_2(x)
        p = self.m(x)


    
        return x, p, attn_weights