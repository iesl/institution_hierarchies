
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable

from main.base_models.OrderedEmb import OrderedEmb
from main.base_models.embs.LSTMEmb import LSTMEmb


'''
SetLSTM first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with an LSTM. Finally, it it scores these embeddings with dot product as a similairty measure  
'''
class OrderedLSTM(OrderedEmb):
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens
        """
        super(OrderedLSTM, self).__init__(config, data_extractor, tokenizer)


        self.emb_counter = 0

        self.inst_emb = LSTMEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.inst_emb_dim, self.config.inst_rnn_dim, self.config.inst_lstm_bidirectional, \
                                                2 if self.config.inst_lstm_bidirectional == True else 1)
        self.list_emb = [self.inst_emb]
        self.emb_counter += 1

        if self.config.include_city:
            self.city_emb = LSTMEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.city_emb_dim, self.config.city_rnn_dim, self.config.city_lstm_bidirectional,\
                                        2 if self.config.city_lstm_bidirectional == True else 1)
            self.list_emb.append(self.city_emb)
            self.emb_counter += 1

        if self.config.include_state:
            self.state_emb = LSTMEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.state_emb_dim, self.config.state_rnn_dim, self.config.state_lstm_bidirectional,\
                                        2 if self.config.state_lstm_bidirectional == True else 1)
            self.list_emb.append(self.state_emb)
            self.emb_counter += 1

        if self.config.include_country:
            self.country_emb = LSTMEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.country_emb_dim, self.config.country_rnn_dim, self.config.country_lstm_bidirectional, \
                                       2 if self.config.country_lstm_bidirectional == True else 1)
            self.list_emb.append(self.country_emb)
            self.emb_counter +=1

        if self.config.include_type:
            self.type_emb = LSTMEmb(self.dict_vocab[self.emb_counter].size + 1, self.config.type_emb_dim, self.config.type_rnn_dim, self.config.type_lstm_bidirectional, \
                                       2 if self.config.type_lstm_bidirectional == True else 1)
            self.list_emb.append(self.type_emb)
            self.emb_counter +=1


    def compute_loss(self, batch_ids):
        """
        Computes loss for batch of query positive negative triplets

        param qry_tk: query tokens (batch size of list of tokens)
        param pos_tk: positive mention lookup (batch size of list of tokens)
        param neg_tk: negative mention lookup (batch size of list of tokens)
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
