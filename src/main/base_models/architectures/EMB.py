import torch
import torch.nn as nn
from torch.autograd import Variable

from main.utils.util import device


"""
Embedding building block class
"""
class EMB(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, init_emb=None):
        '''
        param vocab_size: size of vocabulary
        param emebeding_dim: embedding dimension
        '''

        super(EMB, self).__init__()

        if init_emb is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(init_emb)


    def forward(self, string_lkup):
        """
        Looks up the embedding for the string lookup integers 
        
        param string_lkup: lookup ints for string
        return emb: embeddings matrix 
        return mask: mask for embeddings 
        """
        mask = Variable((string_lkup > 0).float()).to(device)
        emb = self.embedding(string_lkup)
        return emb, mask
