from main.base_models.embs.TransformerPosEmb import TransformerPosEmb
from main.base_models.OrderedEmb import OrderedEmb


'''
SetTransformer first gets the set difference and set intersection between the query and candidate. Then, it embeds these sets 
with a Transformer. Finally, it it scores these embeddings with dot product as a similairty measure  
'''
class OrderedTransformer(OrderedEmb):
    def __init__(self, config, data_extractor, tokenizer):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens
        """
        super(OrderedTransformer, self).__init__(config, data_extractor, tokenizer)

        self.emb_counter = 0
        self.inst_emb = TransformerPosEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.inst_trans_num_layers), int(self.config.inst_emb_dim), int(self.config.inst_ff_dim), \
                        int(self.config.inst_num_heads), float(config.dropout_rate))
        self.list_emb = [self.inst_emb]
        self.emb_counter += 1

        if self.config.include_city:
            self.city_emb = TransformerPosEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.city_trans_num_layers), int(self.config.city_emb_dim),
                                               int(self.config.city_ff_dim), int(self.config.city_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.city_emb)
            self.emb_counter += 1

        if self.config.include_state:
            self.state_emb = TransformerPosEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.state_trans_num_layers), int(self.config.state_emb_dim),
                                               int(self.config.state_ff_dim),  int(self.config.state_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.state_emb)
            self.emb_counter += 1

        if self.config.include_country:
            self.country_emb = TransformerPosEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.country_trans_num_layers), int(self.config.country_emb_dim),
                                               int(self.config.country_ff_dim),  int(self.config.country_num_heads), float(config.dropout_rate))
            self.list_emb.append(self.country_emb)
            self.emb_counter +=1

        if self.config.include_type:
            self.type_emb = TransformerPosEmb(self.dict_vocab[self.emb_counter].size + 1, int(self.config.type_trans_num_layers), int(self.config.type_emb_dim),
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


    def score_dev_test_batch(self, batch_ids):
        """
        Returns the score for query candidate pair

        param qry_tk: query mention lookup (batch size of list of tokens)
        param cnd_tk: candidate mention lookup (batch size of list of tokens)
        return: score (batch_size)
        """

        return super().score_dev_test_batch(batch_ids, self.list_emb)
