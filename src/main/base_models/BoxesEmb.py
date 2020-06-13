import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math

from main.base_models.BaseTok import BaseTok

from main.base_models.embs.BoxEmb import BoxEmb

from main.utils.util import device

'''
SetAvgEmb first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with an embedding layer. Finally, it it scores these embeddings with dot product as a similairty measure  
'''

class BoxesEmb(BaseTok):
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens
        """
        super(BoxesEmb, self).__init__(config)
        self.config = config
        self.data_extractor = data_extractor
        self.tokenizer = tokenizer

        self.dict_vocab = super().get_vocab()

        self.inst_linear = nn.Linear(self.config.inst_emb_dim, 1)
        self.list_linear = [self.inst_linear]
        self.counter = 1

        if self.config.include_city:
            self.city_linear = nn.Linear(self.config.city_emb_dim, 1)
            self.list_linear.append(self.city_linear)
            self.counter += 1

        if self.config.include_state:
            self.state_linear = nn.Linear(self.config.state_emb_dim, 1)
            self.list_linear.append(self.state_linear)
            self.counter += 1

        if self.config.include_country:
            self.country_linear = nn.Linear(self.config.country_emb_dim, 1)
            self.list_linear.append(self.country_linear)
            self.counter +=1

        if self.config.include_type:
            self.type_linear = nn.Linear(self.config.type_emb_dim, 1)
            self.list_linear.append(self.type_linear)
            self.counter +=1

        self.weighted_sum_score = nn.Linear(self.counter, 1)

        # Vector of ones (used for loss)
        self.ones = Variable(torch.ones(config.train_batch_size, 1).to(device))
        self.loss = BCEWithLogitsLoss()

    def get_counter(self):
        '''
        Get the number of types to include

        :return:
        '''
        return self.counter

    def compute_loss(self, batch_ids, list_emb):
        """
        Computes loss for batch of query positive negative triplets

        param batch_ids: (qry_id, pos_id, neg_id)
        return: loss (batch_size)
        """

        batch_data = self.data_extractor.get_batch_data(batch_ids)
        batch_tk = self.tokenizer.tokenize_batch_data(batch_data)

        batch_set_tok = super().forward(batch_tk) # [num_fields, num_set_op, batch_size, emb_dim]

        list_pos_score = []
        list_neg_score = []

        for idx, field_set_tok in enumerate(batch_set_tok):
            qry_emb, qry_mask, pos_emb, pos_mask, neg_emb, neg_mask = list_emb[idx](field_set_tok, need_mask=True)

            pos_score = self.score_pair_train(qry_emb, qry_mask, pos_emb, pos_mask, self.list_linear[idx])
            neg_score = self.score_pair_train(qry_emb, qry_mask, neg_emb, neg_mask, self.list_linear[idx])

            list_pos_score.append(pos_score)
            list_neg_score.append(neg_score)

        pos_score = self.weighted_sum_score(torch.stack(list_pos_score, dim=1).to(device))
        neg_score = self.weighted_sum_score(torch.stack(list_neg_score, dim=1).to(device))

        loss = self.loss((pos_score - neg_score), self.ones)
        return loss

    def score_pair_train(self, qry_emb, qry_mask, cnd_emb, cnd_mask, linear_layer):
        """
        Scores the batch of query candidate pair
        Take the dot product of set representations and then a weighted sum of the scores

        param qry_emb: query mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd_emb: candidate mention embedding (batch_size * max_len_token * hidden_state_output_size)
        return: score for query candidate pairs (batch_size * 1)
        """

        # TODO: normalize based on size of qry or cnd

        qry_len = torch.sum(qry_mask, dim=1)
        cnd_len = torch.sum(cnd_mask, dim=1)


        qry_cnd_insct, qry_cnd_mask, neg_qry_cnd_mask = self.intersect_emb(qry_emb, qry_mask, cnd_emb, cnd_mask)
        qry_cnd_vol = self.get_total_volume(qry_cnd_insct, qry_cnd_mask, neg_qry_cnd_mask, qry_len, cnd_len)

        qry_insct, qry_qry_mask, neg_qry_qry_mask = self.intersect_emb(qry_emb, qry_mask, qry_emb, qry_mask)
        qry_vol = self.get_total_volume(qry_insct, qry_qry_mask, neg_qry_qry_mask, qry_len, qry_len)

        cnd_insct, cnd_cnd_mask, neg_cnd_cnd_mask = self.intersect_emb(cnd_emb, cnd_mask, cnd_emb, cnd_mask)
        cnd_vol = self.get_total_volume(cnd_insct, cnd_cnd_mask, neg_cnd_cnd_mask, cnd_len, cnd_len)

        cnd_sim = qry_cnd_vol / cnd_vol
        qry_sim = qry_cnd_vol / qry_vol

        cnd_diff = 1 - cnd_sim

        score = cnd_sim - qry_sim - cnd_diff


        return score


    def score_dev_test_batch(self, batch_ids, list_emb):
        """
        Returns the score for query candidate pair

        param qry_tk: query mention lookup (batch size of list of tokens)
        param cnd_tk: candidate mention lookup (batch size of list of tokens)
        return: score (batch_size)
        """

        batch_data = self.data_extractor.get_batch_data(batch_ids, is_train=False)

        batch_tk = self.tokenizer.tokenize_batch_data(batch_data, is_train=False)

        batch_set_tok = super().evaluate(batch_tk)

        list_scores = []

        for idx, field_set_tok in enumerate(batch_set_tok):
            qry_emb, qry_mask, cnd_emb, cnd_mask = list_emb[idx].evaluate(batch_set_tok[idx], need_mask=True)

            score = self.score_pair_train(qry_emb, qry_mask, cnd_emb, cnd_mask, self.list_linear[idx])
            list_scores.append(score)

        score = self.weighted_sum_score(torch.stack(list_scores, dim=1).to(device))

        return score

    def intersect_emb(self, emb_one, emb_one_mask, emb_two, emb_two_mask):
        '''
        Finds the intersection between embeddings

        :param emb_one: # [batch_size, max_len_token, emb_dim/2, 2]
        :param emb_one_mask:  # [batch_size, max_len_token]
        :param emb_two: # [batch_size, max_len_token, emb_dim/2, 2]
        :param emb_two_mask: # [batch_size, max_len_token]
        :return:
        '''



        emb_one_min_max = torch.stack([emb_one[:,:,:,0], (emb_one[:,:,:,0] + emb_one[:,:,:,1])], dim=3)
        emb_two_min_max = torch.stack([emb_two[:,:,:,0], (emb_two[:,:,:,0] + emb_two[:,:,:,1])], dim=3)

        neg_emb_one_mask = torch.ones_like(emb_one_mask) - emb_one_mask
        neg_emb_two_mask = torch.ones_like(emb_two_mask) - emb_two_mask


        emb_one_max_len_token = emb_one_min_max.shape[1]
        emb_two_max_len_token = emb_two_min_max.shape[1]

        neg_emb_one_mask_expanded = neg_emb_one_mask.repeat(1, emb_two_max_len_token)
        neg_emb_two_mask_expanded = torch.repeat_interleave(neg_emb_two_mask, emb_one_max_len_token, dim=1)


        neg_emb_mask = torch.max(neg_emb_one_mask_expanded, neg_emb_two_mask_expanded) # [batch_size, max_len_token^2]


        emb_mask = torch.ones_like(neg_emb_mask) - neg_emb_mask # [batch_size, max_len_token^2]

        emb_one_expanded = emb_one.repeat(1, emb_two_max_len_token, 1, 1) # [batch_size, max_len_token^2, emb_dim/2, 2]
        emb_two_expanded = torch.repeat_interleave(emb_two, emb_one_max_len_token, dim=1)  # [batch_size, max_len_token^2, emb_dim/2, 2]

        max_val = torch.max(emb_one_expanded, emb_two_expanded)  # [batch_size, max_len_token^2, emb_dim/2, 2]
        min_val = torch.min(emb_one_expanded, emb_two_expanded)  # [batch_size, max_len_token^2, emb_dim/2, 2]

        mask_keep_zero_index = torch.zeros_like(max_val).byte().to(device)  # [batch_size, max_len_token^2, emb_dim/2, 2]
        mask_keep_zero_index[:,:,:,0] = 1
        mask_keep_first_index = ~mask_keep_zero_index   # [batch_size, max_len_token^2, emb_dim/2, 2]


        intsct = mask_keep_zero_index.float() * max_val + mask_keep_first_index.float() * min_val # [batch_size, max_len_token^2, emb_dim/2, 2]

        return intsct, emb_mask, neg_emb_mask

    def get_total_volume(self, emb, mask, neg_mask, qry_len, cnd_len):
        '''
        Gets the total volume in all the boxes

        :param emb: [batch_size, max_len_token^2, emb_dim/2, 2]
        :param mask: [batch_size, max_len_token^2]
        :param neg_mask: [batch_size, max_len_token^2]
        :return:
        '''

        axis_dim = emb[:,:,:,1] - emb[:,:,:,0]  # [batch_size, max_len_token^2, emb_dim/2]
        axis_dim = torch.log(1 + torch.exp(axis_dim))

        axis_dim_masked = axis_dim * mask[:,:,None] + neg_mask[:,:,None]

        prod_out_emb = torch.prod(axis_dim_masked, dim=2)

        sum_out_max_len = torch.sum(prod_out_emb, dim=1) / (qry_len * cnd_len)

        return sum_out_max_len




