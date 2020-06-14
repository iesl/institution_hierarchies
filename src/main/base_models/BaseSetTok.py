
from main.objects.Vocab import Vocab
from main.utils.token_lookup import get_qry_pos_neg_insct_min_tok_lookup, get_qry_cnd_insct_min_tok_lookup

import os
import numpy as np
import torch


class BaseSetTok(torch.nn.Module):
    def __init__(self, config):
        """
        param config: config object
        """
        super(BaseSetTok, self).__init__()
        self.config = config

        list_include_flags = [True, self.config.include_city, self.config.include_state, self.config.include_country, self.config.include_type]
        list_tokenizer = [config.inst_tokenizer_name, config.city_tokenizer_name, config.state_tokenizer_name, config.country_tokenizer_name, config.type_tokenizer_name]

        # Repeat state twice since state and country have same config
        list_max_len_token = [(config.max_num_inst_char, config.max_num_inst_unigram), (config.max_num_city_char, config.max_num_city_unigram), \
                              (config.max_num_state_char, config.max_num_state_unigram), (config.max_num_state_char, config.max_num_state_unigram),\
                              (config.max_num_type_char, config.max_num_type_unigram)]

        self.dict_vocab = {}

        for idx, flag in enumerate(list_include_flags):

            if flag:
                tokenizer_name =  list_tokenizer[idx]

                if tokenizer_name == "Char":
                    max_len_token = list_max_len_token[idx][0]
                else:
                    max_len_token = list_max_len_token[idx][1]

                if self.config.fold is not None:
                    vocab_file = os.path.join("data", "cross_validation", "fold_%d" % self.config.fold, "vocab_"+ tokenizer_name.lower())
                else:
                    vocab_file = os.path.join("data", "orig_dataset", "vocab_"+ tokenizer_name.lower())

                vocab = Vocab(vocab_file, max_len_token)

                self.dict_vocab[len(self.dict_vocab)] = vocab

    def get_vocab(self):
        '''
        Returns all the vocab used by the tokenizer
        :return:
        '''

        return self.dict_vocab

    def forward(self, batch_tk):
        '''
        Forward pass that does set operation on tokens during training

        :param batch_tok: batch of token (qry_tk, pos_tk, neg_tk)
        :param is_train:
        :return:
        '''

        (qry_tk, pos_tk, neg_tk) = list(zip(*batch_tk))

        qry_tk = np.asarray(qry_tk)
        pos_tk = np.asarray(pos_tk)
        neg_tk = np.asarray(neg_tk)

        batch_tk = []

        for (idx, vocab) in self.dict_vocab.items():
            qry_lkup, pos_lkup, neg_lkup, qry_min_pos_lkup, pos_min_qry_lkup, qry_insct_pos_lkup, qry_min_neg_lkup, neg_min_qry_lkup, qry_insct_neg_lkup = \
                get_qry_pos_neg_insct_min_tok_lookup(vocab, qry_tk[:,idx], pos_tk[:,idx], neg_tk[:,idx])

            batch_tk.append((qry_lkup, pos_lkup, neg_lkup, qry_min_pos_lkup, pos_min_qry_lkup, qry_insct_pos_lkup, qry_min_neg_lkup, neg_min_qry_lkup, qry_insct_neg_lkup))

        return batch_tk

    def evaluate(self, batch_tk):
        '''
        Forward pass that does set operation on tokens during eval

        :param batch_tok: batch of token  (qry_tk, cnd_tk, lbl)
        :param is_train:
        :return:
        '''

        (qry_tk, cnd_tk) = list(zip(*batch_tk))

        qry_tk = np.asarray(qry_tk)
        cnd_tk = np.asarray(cnd_tk)

        batch_tk = []
        for (idx, vocab) in self.dict_vocab.items():
            qry_lkup, cnd_lkup, qry_min_cnd_lkup, cnd_min_qry_lkup, qry_insct_cnd_lkup = get_qry_cnd_insct_min_tok_lookup(vocab, qry_tk[:,idx], cnd_tk[:,idx])
            batch_tk.append((qry_lkup, cnd_lkup, qry_min_cnd_lkup, cnd_min_qry_lkup, qry_insct_cnd_lkup))

        return batch_tk