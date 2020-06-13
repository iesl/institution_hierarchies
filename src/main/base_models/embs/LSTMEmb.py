import torch
import torch.nn as nn
import numpy as np

from main.base_models.embs.OrderedFieldEmb import OrderedFieldEmb
from main.base_models.architectures.EMB import EMB

from main.utils.util import device


class LSTMEmb(OrderedFieldEmb):

    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size, is_bidirectional, num_directions):
        '''
        param emb: embedding class
        param embedding_dim: embedding dimension
        param rnn_hidden_size: rnn_hidden_size dimension
        param is_bidirectional: is the rnn bidirectional
        '''
        super(OrderedFieldEmb, self).__init__()

        self.emb = EMB(vocab_size, embedding_dim)
        self.output_dim = rnn_hidden_size * num_directions

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers=1, bidirectional=is_bidirectional, batch_first=True)

    def embed(self, set_lkup):
        '''
        Embed using average string embedding

        :param set_lkup:
        :return:
        '''

        string_emb, mask = self.emb(torch.from_numpy(np.asarray(set_lkup)).to(device)) # ( [batch_size, max_len_token, emb_dim], [batch_size, max_len_token])
        string_emb, final_hn_cn = self.lstm(string_emb) # [batch_size, max_len_token, output_dim]

        string_len = torch.sum(mask, dim=1).view(-1, 1).unsqueeze(2).repeat(1, 1, self.output_dim).long() - 1 # [batch_size, 1, output_dim]
        string_len = torch.max(string_len, torch.zeros_like(string_len))

        last_idx_string_emb = torch.gather(input=string_emb, dim=1, index=string_len)

        return last_idx_string_emb.squeeze(1)

    def get_output_dim(self):
        '''
        Get output dim calculated from rnn_hidden_size and bidirectional or not

        :return:
        '''
        return self.output_dim



