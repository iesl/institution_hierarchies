import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math
import sys

from main.base_models.embs.TransformerEmb import TransformerEmb
from main.base_models.SetEmb import SetEmb


'''
SetTransformer first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with a Transformer. Finally, it it scores these embeddings with dot product as a similairty measure  
'''
class SetTransformer(SetEmb):
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens 
        """
        super(SetTransformer, self).__init__(config, data_extractor, tokenizer)

        self.emb_counter = 0
        self.inst_emb = TransformerEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.inst_trans_num_layers), int(self.config.inst_emb_dim), int(self.config.inst_ff_dim), \
                        int(self.config.inst_num_heads), float(config.dropout_rate))
        self.list_emb = [self.inst_emb]
        self.emb_counter += 1

        if self.config.lin_comb:
            self.lin_comb = nn.Linear(3, 1)

        if self.config.init_lin_comb:
            self.lin_comb.weight.data.copy_(torch.from_numpy(np.asarray([1, -1, -1])))
            self.lin_comb.bias.data.copy_(torch.zeros(1))

        if self.config.mlp:
            self.mlp_lin_1 = nn.Linear(3, 27)
            self.relu = nn.ReLU()
            self.mlp_lin_2 = nn.Linear(27, 1)


        if self.config.include_city:
            self.city_emb = TransformerEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.city_trans_num_layers), int(self.config.city_emb_dim),
                                               int(self.config.city_ff_dim), int(self.config.city_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.city_emb)
            self.emb_counter += 1

        if self.config.include_state:
            self.state_emb = TransformerEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.state_trans_num_layers), int(self.config.state_emb_dim),
                                               int(self.config.state_ff_dim),  int(self.config.state_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.state_emb)
            self.emb_counter += 1

        if self.config.include_country:
            self.country_emb = TransformerEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.country_trans_num_layers), int(self.config.country_emb_dim),
                                               int(self.config.country_ff_dim),  int(self.config.country_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.country_emb)
            self.emb_counter +=1

        if self.config.include_type:
            self.type_emb = TransformerEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.type_trans_num_layers), int(self.config.type_emb_dim),
                                               int(self.config.country_ff_dim),  int(self.config.type_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.type_emb)
            self.emb_counter +=1

        assert(super().get_counter() == self.counter)


    def compute_loss(self, batch_ids):
        """
        Computes loss for batch of query positive negative triplets

        param qry_tk: query tokens (batch size of list of tokens)
        param pos_tk: positive mention lookup (batch size of list of tokens)
        param neg_tk: negative mention lookup (batch size of list of tokens)
        return: loss (batch_size)
        """
        return super().compute_loss(batch_ids, self.list_emb)



    def score_pair_train(self, qry_emb, cnd_emb, qry_min_cnd_emb, cnd_min_qry_emb, qry_insct_cnd_emb, linear_layer):
        """
        Scores the batch of query candidate pair
        Take the dot product of set representations and then a weighted sum of the scores

        param qry_emb: query mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd_emb: candidate mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param qry_min_cnd_emb: query set difference candidate embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd_min_qry_emb: candidate set difference query embedding (batch_size * max_len_token * hidden_state_output_size)
        param qry_insct_cnd_emb: query set intersection candidate embedding (batch_size * max_len_token * hidden_state_output_size)
        return: score for query candidate pairs (batch_size * 1)
        """
        qry_sim = qry_emb * qry_insct_cnd_emb
        cnd_sim = cnd_emb * qry_insct_cnd_emb
        cnd_diff = cnd_emb * cnd_min_qry_emb

        qry_sim_score = linear_layer(qry_sim)
        cnd_sim_score = linear_layer(cnd_sim)
        cnd_diff_score = linear_layer(cnd_diff)

        if self.config.lin_comb:
            concat_sim = torch.stack([qry_sim_score, cnd_sim_score, cnd_diff_score], dim=2)
            output = self.lin_comb(concat_sim).squeeze(2)

        elif self.config.mlp:
            concat_sim = torch.stack([qry_sim_score, cnd_sim_score, cnd_diff_score], dim=2)
            output = self.mlp_lin_2(self.relu(self.mlp_lin_1(concat_sim))).squeeze(2)
        else:
            output = qry_sim_score - cnd_sim_score - cnd_diff_score
        return output.squeeze(1)


    def score_dev_test_batch(self, batch_ids):
        """
        Returns the score for query candidate pair

        param qry_tk: query mention lookup (batch size of list of tokens)
        param cnd_tk: candidate mention lookup (batch size of list of tokens)
        return: score (batch_size)
        """

        return super().score_dev_test_batch(batch_ids, self.list_emb)
