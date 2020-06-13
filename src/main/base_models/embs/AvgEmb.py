import torch
import numpy as np

from main.base_models.architectures.EMB import EMB
from main.base_models.embs.SetFieldEmb import SetFieldEmb

from main.utils.util import device


class AvgEmb(SetFieldEmb):

    def __init__(self, vocab_size, embedding_dim):
        super(AvgEmb, self).__init__()
        self.EMB = EMB(vocab_size, embedding_dim)

    def embed(self, set_lkup):
        '''
        Embed using average string embedding

        :param set_lkup:
        :return:
        '''

        string_emb, string_mask = self.EMB(torch.from_numpy(np.asarray(set_lkup)).to(device) )# ( [batch_size, max_len_token, emb_dim], [batch_size, max_len_token])

        string_emb_masked = string_emb * string_mask.unsqueeze(2) #  # [batch_size, max_len_token, emb_dim]

        sum_string_emb = torch.sum(string_emb_masked, dim =1)
        string_len = torch.max(torch.sum(string_mask, dim=1, keepdim=True), torch.ones(string_mask.shape[0], 1).to(device))
        avg_string_emb = sum_string_emb / string_len

        return avg_string_emb


