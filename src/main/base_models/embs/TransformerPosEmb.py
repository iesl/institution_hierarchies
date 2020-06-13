import torch
import numpy as np

from main.base_models.embs.OrderedFieldEmb import OrderedFieldEmb
from main.base_models.architectures.TransformerEncoder import TransformerEncoderPosition

from main.utils.util import device


class TransformerPosEmb(OrderedFieldEmb):

    def __init__(self, vocab_size, num_layers, model_dim, ff_dim, num_heads, dropout_rate):
        '''
        param vocab_size: config object
        param num_layers:
        param model_dim:
        param ff_dim:
        param num_heads:
        param dropout_rate:
        '''
        super(TransformerPosEmb, self).__init__()
        self.transformer = TransformerEncoderPosition(vocab_size, int(num_layers), int(model_dim), int(ff_dim), \
                        int(num_heads), float(dropout_rate))

    def embed(self, set_lkup):
        '''
        Embed using average string embedding

        :param set_lkup:
        :return: transformer emb
        :return: transformer mask
        '''
        set_lkup = torch.from_numpy(np.asarray(set_lkup)).to(device)
        string_emb, string_mask = self.transformer(set_lkup)

        string_mask = string_mask.squeeze(1)

        string_emb_masked = string_emb * string_mask.unsqueeze(2) #  # [batch_size, max_len_token, emb_dim]

        sum_string_emb = torch.sum(string_emb_masked, dim =1)
        string_len = torch.max(torch.sum(string_mask, dim=1, keepdim=True), torch.ones(string_mask.shape[0], 1).to(device))
        avg_string_emb = sum_string_emb / string_len

        return avg_string_emb



