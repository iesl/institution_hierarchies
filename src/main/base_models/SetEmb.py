import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math

from main.base_models.BaseSetTok import BaseSetTok

from main.utils.util import device


'''
SetAvgEmb first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with an embedding layer. Finally, it it scores these embeddings with dot product as a similairty measure  
'''
class SetEmb(BaseSetTok):
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens
        """
        super(SetEmb, self).__init__(config)
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
        self.ones = Variable(torch.ones(config.train_batch_size, 1)).to(device)
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
            qry_emb, pos_emb, neg_emb, qry_min_pos_emb, pos_min_qry_emb, qry_insct_pos_emb, qry_min_neg_emb, \
            neg_min_qry_emb, qry_insct_neg_emb = list_emb[idx](field_set_tok)

            pos_score = self.score_pair_train(qry_emb, pos_emb, qry_min_pos_emb, pos_min_qry_emb, qry_insct_pos_emb, self.list_linear[idx])
            neg_score = self.score_pair_train(qry_emb, neg_emb, qry_min_neg_emb, neg_min_qry_emb, qry_insct_neg_emb, self.list_linear[idx])

            list_pos_score.append(pos_score)
            list_neg_score.append(neg_score)

        pos_score = self.weighted_sum_score(torch.stack(list_pos_score, dim=1).to(device))
        neg_score = self.weighted_sum_score(torch.stack(list_neg_score, dim=1).to(device))

        loss = self.loss(pos_score - neg_score, self.ones)
        return loss

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
        # Represents how much of parent is in child
        cnd_sim = cnd_emb * qry_insct_cnd_emb
        # Represents how much of child is in parent
        qry_sim = qry_emb * qry_insct_cnd_emb
        # Represents how much of parent not in child is similar to parent
        cnd_diff = cnd_emb * cnd_min_qry_emb

        output = cnd_sim - qry_sim - cnd_diff
        scores = linear_layer(output)

        return scores.squeeze(1)


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
            qry_emb, cnd_emb, qry_min_cnd_emb, cnd_min_qry_emb, qry_insct_cnd_emb = list_emb[idx].evaluate(
                batch_set_tok[idx])

            score = self.score_pair_train(qry_emb, cnd_emb, qry_min_cnd_emb, cnd_min_qry_emb, qry_insct_cnd_emb, self.list_linear[idx])
            list_scores.append(score)

        score = self.weighted_sum_score(torch.stack(list_scores, dim=1).to(device))

        return score