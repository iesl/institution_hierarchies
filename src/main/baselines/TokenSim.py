import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math

from main.base_models.BaseSetTok import BaseSetTok

from main.utils.token_lookup import get_qry_cnd_tok


'''
TokenSim first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with an embedding layer. Finally, it it scores these embeddings with dot product as a similairty measure  
'''
class TokenSim():
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens
        """
        super(TokenSim, self).__init__()
        self.config = config
        self.data_extractor = data_extractor
        self.tokenizer = tokenizer

        list_include_flags = [True, self.config.include_city, self.config.include_state, self.config.include_country, self.config.include_type]
        self.counter = 0

        for flag in list_include_flags:
            if flag:
                self.counter += 1

    def score_pair_train(self, qry_tk, cnd_tk, qry_min_cnd_tk, cnd_min_qry_tk, qry_insct_cnd_tk):
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

        list_scores = []

        for num, denom in zip(qry_insct_cnd_tk, cnd_tk):

            if len(denom) == 0:
                list_scores.append(0)
            else:
                list_scores.append(len(num) / len(denom))

        return list_scores


    def score_dev_test_batch(self, batch_ids):
        """
        Returns the score for query candidate pair

        param qry_tk: query mention lookup (batch size of list of tokens)
        param cnd_tk: candidate mention lookup (batch size of list of tokens)
        return: score (batch_size)
        """

        batch_data = self.data_extractor.get_batch_data(batch_ids, is_train=False)

        batch_tk = self.tokenizer.tokenize_batch_data(batch_data, is_train=False)

        (batch_qry_tk, batch_cnd_tk) = list(zip(*batch_tk))
        batch_qry_tk = np.asarray(batch_qry_tk)
        batch_cnd_tk = np.asarray(batch_cnd_tk)

        list_scores = []

        for idx in range(self.counter):
            qry_tk, cnd_tk, qry_min_cnd_tk, cnd_min_qry_tk, qry_insct_cnd_tk = get_qry_cnd_tok(batch_qry_tk[:,idx], batch_cnd_tk[:,idx])
            score = self.score_pair_train(qry_tk, cnd_tk, qry_min_cnd_tk, cnd_min_qry_tk, qry_insct_cnd_tk)
            list_scores.append(score)

        list_scores = np.array(list_scores)

        score = np.sum(list_scores, axis=0)

        return score