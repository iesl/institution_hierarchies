import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math

from main.base_models.SetEmb import SetEmb
from main.base_models.embs.AvgEmb import AvgEmb


'''
SetAvgEmb first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with an embedding layer. Finally, it it scores these embeddings with dot product as a similairty measure  
'''
class SetAvgEmb(SetEmb):
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens 
        """
        super(SetAvgEmb, self).__init__(config, data_extractor, tokenizer)

        self.emb_counter = 0
        self.inst_emb = AvgEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.inst_emb_dim)
        self.list_emb = [self.inst_emb]
        self.emb_counter += 1

        if self.config.include_city:
            self.city_emb = AvgEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.city_emb_dim)
            self.list_emb.append(self.city_emb)
            self.emb_counter += 1

        if self.config.include_state:
            self.state_emb = AvgEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.state_emb_dim)
            self.list_emb.append(self.state_emb)
            self.emb_counter += 1

        if self.config.include_country:
            self.country_emb = AvgEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.country_emb_dim)
            self.list_emb.append(self.country_emb)
            self.emb_counter +=1

        if self.config.include_type:
            self.type_emb = AvgEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.type_emb_dim)
            self.list_emb.append(self.type_emb)
            self.emb_counter +=1

        assert(super().get_counter() == self.emb_counter)


    def compute_loss(self, batch_ids):
        """
        Computes loss for batch of query positive negative triplets

        param batch_ids: (qry_id, pos_id, neg_id)
        return: loss (batch_size)
        """

        return super().compute_loss(batch_ids, self.list_emb)


    def score_dev_test_batch(self, batch_ids):
        """ 
        Returns the score for query candidate pair 

        param qry_tk: query mention lookup (batch size of list of tokens)
        param cnd_tk: candidate mention lookup (batch size of list of tokens)
        return: score (batch_size)
        """
        return super().score_dev_test_batch(batch_ids, self.list_emb)
