import codecs
import numpy as np

'''
Vocab object that represents the vocabulary 
'''
class Vocab(object):
    def __init__(self, filename, max_len_token):
        '''
        param filename: filename of vocab 
        param tokenizer: tokenizer used to generate vocab 
        param max_len_token: maximum number of tokens 
        '''
        self.filename = filename
        self.OOV = "<OOV>"
        self.OOV_INDEX = 1
        (self.token2id, self.id2token) = self.load(self.filename)
        self.PADDING_INDEX = 0
        self.max_len_token = int(max_len_token)
        self.size = len(self.token2id)

    def __len__(self):
        '''
        return: vocab size
        '''
        return self.size

    def load(self, filename):
        token2id = dict()
        id2token = dict()

        token2id[self.OOV] = self.OOV_INDEX
        id2token[self.OOV_INDEX] = self.OOV

        with codecs.open(filename,'r','UTF-8') as fin:
            for line in fin:
                splt = line.split("\t")
                item = splt[0]
                id = int(splt[1].strip())
                token2id[item] = id
                id2token[id] = item

        return token2id, id2token 

    def to_ints(self, list_tokens):
        '''
        Loads the vocab from file 
        
        param filename: file name of vocab 
        return: dictionary of token to id of token 
        return: dictionary of id to token 
        '''
        list_ints = []

        for token in list_tokens:
            list_ints.append(self.token2id.get(token, self.OOV_INDEX))

        if len(list_ints) > self.max_len_token:
            return np.asarray(list_ints[0:self.max_len_token])
        
        # Pad the list of ints if less than max_len
        while len(list_ints) < self.max_len_token:
            list_ints += [self.PADDING_INDEX]

        return np.asarray(list_ints)


