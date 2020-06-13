
import torch


"""
FieldEmb is a superclass for all embedding methods 
"""
class OrderedFieldEmb(torch.nn.Module):

    def __init__(self):
        super(OrderedFieldEmb, self).__init__()


    def forward(self, batch_data, is_dual=False, need_mask=False):
        """
        Looks up the embedding for the institution tokens during training

        :param batch_data: batch of tokens [num_set_op, batch_size, emb_dim]
        :return:
        """
        qry_lkup, pos_lkup, neg_lkup = batch_data

        if is_dual:
            qry_p_emb, pos_emb = self.embed(qry_lkup, pos_lkup)
            qry_n_emb, neg_emb = self.embed(pos_lkup, neg_lkup)

            return qry_p_emb, pos_emb, qry_n_emb, neg_emb

        else:

            if need_mask:
                qry_emb, qry_mask = self.embed(qry_lkup)
                pos_emb, pos_mask = self.embed(pos_lkup)
                neg_emb, neg_mask = self.embed(neg_lkup)
                return qry_emb, qry_mask, pos_emb, pos_mask, neg_emb, neg_mask

            else:
                qry_emb = self.embed(qry_lkup)
                pos_emb = self.embed(pos_lkup)
                neg_emb = self.embed(neg_lkup)
                return qry_emb, pos_emb, neg_emb


    def evaluate(self, batch_data, is_dual=False, need_mask=False):
        '''
        Looks up the embeddings for the tokens during eval

        :param batch_data: batch of tokens [num_set_op, batch_size, emb_dim]
        :return:
        '''
        qry_lkup, cnd_lkup = batch_data

        if is_dual:
            qry_emb, cnd_emb = self.embed(qry_lkup, cnd_lkup)
            return qry_emb, cnd_emb

        else:
            if need_mask:
                qry_emb, qry_mask = self.embed(qry_lkup)
                cnd_emb, cnd_mask = self.embed(cnd_lkup)

                return qry_emb, qry_mask, cnd_emb, cnd_mask

            else:
                qry_emb = self.embed(qry_lkup)
                cnd_emb = self.embed(cnd_lkup)

                return qry_emb, cnd_emb

