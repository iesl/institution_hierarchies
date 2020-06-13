
import torch


"""
FieldEmb is a superclass for all embedding methods 
"""
class SetFieldEmb(torch.nn.Module):

    def __init__(self):
        super(SetFieldEmb, self).__init__()


    def forward(self, batch_data, is_dual=False):
        """
        Looks up the embedding for the institution tokens during training

        :param batch_data: batch of tokens [num_set_op, batch_size, emb_dim]
        :return:
        """
        qry_lkup, pos_lkup, neg_lkup, qry_min_pos_lkup, pos_min_qry_lkup, qry_insct_pos_lkup, qry_min_neg_lkup, neg_min_qry_lkup, qry_insct_neg_lkup = batch_data

        if is_dual:
            qry_p_emb, pos_emb = self.embed(qry_lkup, pos_lkup)
            qry_n_emb, neg_emb = self.embed(pos_lkup, neg_lkup)

            return qry_p_emb, pos_emb, qry_n_emb, neg_emb

        else:
            qry_emb = self.embed(qry_lkup)
            pos_emb = self.embed(pos_lkup)
            neg_emb = self.embed(neg_lkup)

            qry_min_pos_emb = self.embed(qry_min_pos_lkup)
            pos_min_qry_emb = self.embed(pos_min_qry_lkup)
            qry_insct_pos_emb = self.embed(qry_insct_pos_lkup)
            qry_min_neg_emb = self.embed(qry_min_neg_lkup)
            neg_min_qry_emb = self.embed(neg_min_qry_lkup)
            qry_insct_neg_emb = self.embed(qry_insct_neg_lkup)

            return qry_emb, pos_emb, neg_emb, qry_min_pos_emb, pos_min_qry_emb, qry_insct_pos_emb, qry_min_neg_emb, \
                        neg_min_qry_emb, qry_insct_neg_emb

    def evaluate(self, batch_data, is_dual=False):
        '''
        Looks up the embeddings for the tokens during eval

        :param batch_data: batch of tokens [num_set_op, batch_size, emb_dim]
        :return:
        '''
        qry_lkup, cnd_lkup, qry_min_cnd_lkup, cnd_min_qry_lkup, qry_insct_cnd_lkup = batch_data


        if is_dual:
            qry_emb, cnd_emb = self.embed(qry_lkup, cnd_lkup)

            return qry_emb, cnd_emb

        else:
            qry_emb = self.embed(qry_lkup)
            cnd_emb = self.embed(cnd_lkup)
            qry_min_cnd_emb = self.embed(qry_min_cnd_lkup)
            cnd_min_qry_emb = self.embed(cnd_min_qry_lkup)
            qry_insct_cnd_emb = self.embed(qry_insct_cnd_lkup)

            return qry_emb, cnd_emb, qry_min_cnd_emb, cnd_min_qry_emb, qry_insct_cnd_emb

