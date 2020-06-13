import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math
import copy
import torch.nn.functional as F


def clones(module, N):
    '''
    Helper function that clones the module 

    param module: module to clone 
    param N: number of times to clone 
    return: module copied N times
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def get_src_tgt_mask(src, tgt, pad = 0):
    src_mask = (src != pad).unsqueeze(-2)
    # tgt = tgt[:, :-1]
    tgt_mask = make_std_mask(tgt, pad)
    return (src_mask.float(), tgt_mask)

def get_string_mask(src, pad = 0):
    '''
    Helper function that gets the mask for the input 

    param src: src to get the mask for 
    param pad: pad token
    return: source mask 
    '''
    src_mask = (src != pad).unsqueeze(-2)
    return src_mask


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, N, d_model, d_ff, h, dropout):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionFeedForward(d_model, d_ff, dropout)
        position = PositionalEncodings(d_model, dropout)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), 
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), 
            nn.Sequential(Embeddings(d_model, vocab_size+1), c(position)),
            nn.Sequential(Embeddings(d_model, vocab_size+1), c(position)),
            Generator(d_model, vocab_size+1))

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt):
        (src_mask, tgt_mask) = get_src_tgt_mask(src, tgt)
        return self.model(src, src_mask, tgt, tgt_mask)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encode(src, src_mask)
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
'''
Encoder building block class
'''
class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        param layer: layer to use for Encoer 
        param N: number of layers 
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        '''
        Encodes the input using self attention 

        param qry_embs: query embeddings
        param cnd_emb: candidate embedding 
        param qry_mask: query mask 
        param cnd_mask: candidate mask tokens 
        return: encoding for query embeddings and candidate embedding 
        '''
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

'''
LayerNorm building block class that normalizes the input 
'''
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        '''
        param features: number of features in input to normalzie
        param eps: epsilon to ensure stabiltiy in normalizing 
        '''
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        '''
        Normalizes the input 

        params x: tensor to normalize
        return: normalizes the input 
        '''
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 

'''
SubLayerConnection building block class that normalizes the input and applies dropout
'''
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        '''
        param size: number of features in input 
        param dropout: dropout rate
        '''
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        Wrapper for applying skip connection, dropout, and normalization to input

        param x: input tensor 
        param sublayer: sublayer (either self attention or feed foward)
        return: output of sub layer connection 
        ''' 
        return x + self.dropout(sublayer(self.norm(x)))

'''
EncoderLayer building block class that wraps all the necessary tricks together 
'''
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        param size: number of features in input 
        param self_attn: self attention function 
        param fed_forward; feed forward function 
        dropout: dropout rate 
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        '''
        Applies self attention and a feed forward neural network 

        param x: input tensor
        param memmory: source tensor 
        param x_mask: input tensor mask 
        param mem_mask: source tensor mask 
        return: output of 1 layer in the encoder 
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(prog(x), dims = -1)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask = None, dropout=None):
    '''
    Helper function for calculating attention using scaled dot product  

    param query: query tensor 
    param key: key tensor 
    param value: value tensor
    return: attention values 
    '''
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

'''
MultiHeaded Attention building block class
'''
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        '''
        param h: number of heads 
        param d_model: model dimension 
        param dropout: dropout rate
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        '''
        Calculates multiheaded attention and then applies a linear transformation 

        param query: query tensor 
        param key: key tensor 
        param value: value tensor 
        return: attention values 
        '''
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        (query, key, value) = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self_attn = attention(query, key, value, mask = mask, dropout = self.dropout)

        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        return self.linears[-1](x)

'''
Position Feed Forward building block class
'''
class PositionFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        '''
        param d_model: embedding dimension 
        param d_ff: feedforward dimension
        param dropout: dropout rate
        '''
        super(PositionFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Applies a 2 Layer MLP 

        param x: input tensor 
        return: output of feed forward network 
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

'''
Embeddings building block class
'''
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        '''
        param d_model: embedding dimension 
        param vocab: vocab
        '''
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        '''
        Looks up embeddings for x and scales by square root 

        param x: lookup integers x 
        return: embeddings matrix 
        '''
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):
    def __init__(self, d_model, dropout, max_len=200):
        super(PositionalEncodings, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * - (math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        return self.dropout(x)
