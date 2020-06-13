import string
import numpy as np

'''
Character Tokenizer 
'''
class Char():

    def tokenize(self, string):
        '''
        Converts string to list of characters

        param string: string to tokenize
        return: list of characters in string
        '''
        return np.asarray(list(string))

'''
Unigram Tokenizer 
'''
class Unigram():

    def remove_punctuation(self, orig_string):
        '''
        Removes punctuation from string

        param orig_string: string to remove punctuation
        return: string without punctuation
        '''
        string_stripped = orig_string.translate(str.maketrans('','',string.punctuation))
        return string_stripped

    def tokenize(self, orig_string):
        '''
        Converts string to list of unigrams with punctuation removed

        param orig_string: string to tokenize
        return: list of tokens
        '''
        list_unigrams = self.remove_punctuation(orig_string).split(' ')
        return np.asarray(list_unigrams)


'''
Unigram and Upper Case Tokenizer 
'''
class UnigramUC():

    def remove_punctuation(self, orig_string):
        '''
        Removes punctuation from string

        param orig_string: string to remove punctuation
        return: string without punctuation
        '''
        string_stripped = orig_string.translate(str.maketrans('','',string.punctuation))
        return string_stripped

    def tokenize(self, orig_string):
        '''
        Converts string to list of unigrams and upper cases  with punctuation removed

        param orig_string: string to tokenize
        return: list of tokens
        '''
        list_unigrams = self.remove_punctuation(orig_string).split(' ')
        list_unigram_uc = []
        for unigram in list_unigrams:
            if(unigram.isupper()):
                list_unigram_uc.extend(list(unigram))
            else:
                list_unigram_uc.append(unigram)
        return np.asarray(list_unigram_uc)

'''
Tokenizer objects to represent getting different formats of data 
'''
class Tokenizer(object):

    def __init__(self, config):
        '''
        Sets the right flags for tokenizer

        :param config:
        '''
        self.dict_tokenizer = {}

        # List of tokenizer for institution, city, state, and country
        list_tokenizer = [config.inst_tokenizer_name, config.city_tokenizer_name, config.state_tokenizer_name, config.country_tokenizer_name, config.type_tokenizer_name]
        # List of flags to include
        list_include_flags = [True, config.include_city, config.include_state, config.include_country, config.include_type]

        # Looping throough flags and updates stores tokenizer in the data
        for idx, flag in enumerate(list_include_flags):
            if flag:
                tokenizer_name = list_tokenizer[idx]

                if tokenizer_name == "Char":
                    tokenizer = Char()
                elif tokenizer_name == "Unigram":
                    tokenizer = Unigram()
                else:
                    tokenizer = UnigramUC()

                self.dict_tokenizer[len(self.dict_tokenizer)] = tokenizer


    def tokenize_batch_data(self, batch_data, is_train=True):
        '''
        Tokenize data from batch

        :param batch_tok: batch of ids (qry_data, pos_data, neg_data) or (qry_data, cnd_data, lbl)
        :param is_train:
        :return:
        '''
        batch_tk = []

        if is_train:
            for (qry_data, pos_data, neg_data) in batch_data:

                qry_tk = []
                for idx, data in enumerate(qry_data):
                    tokenizer = self.dict_tokenizer[idx]
                    tok_data = tokenizer.tokenize(data)
                    qry_tk.append(tok_data)

                pos_tk = []
                for idx, data in enumerate(pos_data):
                    tokenizer = self.dict_tokenizer[idx]
                    tok_data = tokenizer.tokenize(data)
                    pos_tk.append(tok_data)

                neg_tk = []
                for idx, data in enumerate(neg_data):
                    tokenizer = self.dict_tokenizer[idx]
                    tok_data = tokenizer.tokenize(data)
                    neg_tk.append(tok_data)

                batch_tk.append([qry_tk, pos_tk, neg_tk])

        else:
            for (qry_data, cnd_data) in batch_data:
                qry_tk = []
                for idx, data in enumerate(qry_data):
                    tokenizer = self.dict_tokenizer[idx]
                    tok_data = tokenizer.tokenize(data)
                    qry_tk.append(tok_data)

                cnd_tk = []
                for idx, data in enumerate(cnd_data):
                    tokenizer = self.dict_tokenizer[idx]
                    tok_data = tokenizer.tokenize(data)
                    cnd_tk.append(tok_data)

                batch_tk.append([qry_tk, cnd_tk])

        return batch_tk



