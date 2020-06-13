import torch
import torch.nn as nn
import numpy as np

from main.base_models.embs.OrderedFieldEmb import OrderedFieldEmb
from main.base_models.architectures.EMB import EMB

from main.utils.util import device

class BoxEmb(OrderedFieldEmb):

    def __init__(self, vocab_size, embedding_dim):
        '''
        param emb: embedding class
        param embedding_dim: embedding dimension
        param rnn_hidden_size: rnn_hidden_size dimension
        param is_bidirectional: is the rnn bidirectional
        '''
        super(BoxEmb, self).__init__()

        half_emb_dim = int(embedding_dim / 2)

        min = torch.empty(vocab_size * half_emb_dim).normal_(mean=0, std=0.2)
        delta = torch.log(torch.empty(vocab_size * half_emb_dim).normal_(mean=1, std=0.2))

        init_emb = torch.stack([min, delta], dim=1)
        init_emb = init_emb.reshape(vocab_size, embedding_dim)
        self.emb = EMB(vocab_size, embedding_dim, init_emb=init_emb)

    def embed(self, set_lkup):
        '''
        Embed using average string embedding

        :param set_lkup:
        :return: constrained emb # [batch_size, max_len_token, emb_dim/2, 2]
        '''

        string_emb, mask = self.emb(torch.from_numpy(np.asarray(set_lkup)).to(device)) # ( [batch_size, max_len_token, emb_dim], [batch_size, max_len_token])

        string_constrain_emb = self.constrain(string_emb)

        return string_constrain_emb, mask

    def constrain(self, emb):
        '''
        Ensure that element at index 0 is less than element at index 1

        :param emb: # [batch_size, max_len_token, emb_dim/2, 2]
        :return:
        '''

        (batch_size, max_len_token, emb_dim) = emb.shape
        half_emb_dim = int(emb_dim /2)
        emb_reshape = emb.reshape(batch_size * max_len_token, emb_dim)


        # Using min, max formulation
        # emb_reshape = emb_reshape.reshape(batch_size * max_len_token * half_emb_dim, 2)
        # identity = torch.eye(2).reshape(1, 2, 2).byte().cuda()
        # identity_T = ~identity.reshape(1, 2, 2).cuda()
        #
        # batch_identity = identity.repeat(batch_size * max_len_token * half_emb_dim, 1, 1).float()  # [batch_size * max_len_token * half_emb_dim, 2, 2]
        # batch_identity_T = identity_T.repeat(batch_size * max_len_token * half_emb_dim, 1, 1).float()  # [batch_size * max_len_token * half_emb_dim, 2, 2]
        #
        # min_idx = emb_reshape[:, 0]
        # max_idx = emb_reshape[:, 1]
        #
        # # Keep indices already have min < max
        # keep_index = (max_idx > min_idx)[:, None, None].float() # [batch_size * max_len_token * half_emb_dim, 1, 1]
        # # Indices that have to be flipped have max > min
        # flip_index = (min_idx > max_idx)[:, None, None].float() # [batch_size * max_len_token * half_emb_dim, 1, 1]
        #
        # # Mask to keep correct indices and flip wrong indices
        # mask = keep_index * batch_identity + flip_index * batch_identity_T  # [batch_size * max_len_token * half_emb_dim, 2, 2]
        # emb_const = torch.bmm(emb_reshape.unsqueeze(1), mask).squeeze(1)  # [batch_size * max_len_token * half_emb_dim, 2, 2]
        #
        # return emb_const.reshape(batch_size, max_len_token, half_emb_dim, 2)

        emb_reshape = emb_reshape.reshape(batch_size, max_len_token, half_emb_dim, 2)

        min_val = emb_reshape[:,:,:,0]
        delta_val = torch.exp(emb_reshape[:,:,:,1])

        constrain_emb = torch.stack([min_val, delta_val], dim=3)

        return constrain_emb
