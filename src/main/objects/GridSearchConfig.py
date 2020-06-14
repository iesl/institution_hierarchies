import itertools
import json
import os
import random
import string

from main.objects.Config import Config

'''
GridSearchConfig objects that stores parameters for doing grid search over hyperparameters
'''
class GridSearchConfig(Config):
    def __init__(self,filename=None):
        '''
        Initializes the hyperparameters to default values 
        
        param filename: filename containing the hyperparameters to use for training 
        '''
        super(Config, self).__init__()
        self.grid_search_fields = []

        self.learning_rate = 0.0001
        self.l2penalty = 10.0

        self.num_batches = 10000
        self.eval_every_minibatch = 400
        self.train_batch_size = 32
        self.dev_test_batch_size = 64

        self.max_num_inst_char = 150
        self.max_num_inst_unigram = 25
        self.max_num_city_char = 20
        self.max_num_city_unigram = 5

        self.max_num_state_char = 10
        self.max_num_state_unigram = 4

        self.max_num_type_char = 10
        self.max_num_type_unigram = 2

        self.inst_tokenizer_name = "tokenizer"
        self.city_tokenizer_name = "tokenizer"
        self.state_tokenizer_name = "tokenizer"
        self.country_tokenizer_name = "tokenizer"
        self.type_tokenizer_name = "tokenizer"

        self.inst_emb_dim = 128
        self.city_emb_dim = 64
        self.state_emb_dim = 64
        self.country_emb_dim = 64
        self.type_emb_dim = 64

        self.inst_rnn_dim = 128
        self.city_rnn_dim = 64
        self.state_rnn_dim = 64
        self.country_rnn_dim = 64
        self.type_rnn_dim = 64

        self.inst_lstm_bidirectional = True
        self.city_lstm_bidirectional = True
        self.state_lstm_bidirectional = True
        self.country_lstm_bidirectional = True
        self.type_lstm_bidirectional = True

        self.inst_trans_num_layers = 4
        self.inst_ff_dim = 128
        self.inst_num_heads = 8
        self.city_trans_num_layers = 2
        self.city_ff_dim = 64
        self.city_num_heads = 4
        self.state_trans_num_layers = 2
        self.state_ff_dim = 64
        self.state_num_heads = 4
        self.country_trans_num_layers = 2
        self.country_ff_dim = 64
        self.country_num_heads = 4
        self.type_trans_num_layers = 2
        self.type_ff_dim = 64
        self.type_num_heads = 4

        self.include_city = False
        self.include_state = False
        self.include_country = False
        self.include_type = False

        self.random_seed = 2524

        self.dropout_rate = 0.2
        self.clip = 0.25

        self.model_name = "model"
        self.tokenizer_name = "tokenizer"
        self.random = random.Random(self.random_seed)

        if filename:
            self.__dict__.update(json.load(open(filename)))

        self.update_boolean()

    def to_json(self):
        '''
        Stores all the parameters into a json 
        '''
        res = {}
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is str or type(self.__dict__[k]) is float or type(self.__dict__[k]) is int:
                res[k] = self.__dict__[k]
        return json.dumps(res)

    def save_config(self,exp_dir):
        '''
        Saves the parameters used for training in experiment directory 
        
        param exp_dir: experiment directory to save configuration 
        '''
        with open(os.path.join(exp_dir,"config.json"), 'w') as fout:
            fout.write(self.to_json())
            fout.write("\n")

    def copy_config(self):
        '''
        Copies the config to use for different grid search configurations
        '''
        c = Config()
        c.__dict__ = self.__dict__.copy()
        return c

    def update_boolean(self, ):
        '''
        Update boolean parameter with boolean value and not string boolean
        '''
        for k in self.__dict__.keys():
            if self.__dict__[k] == "False":
                self.__dict__[k] = False
            elif self.__dict__[k] == "True":
                self.__dict__[k] = True

    def configs_iter(self):
        '''
        Iterates through the grid search fields and creates a configuration for each possible configuration of grid search paramter values 
        
        return: grid search configuration 
        '''
        gs_vals = [self.__dict__[k] for k in self.grid_search_fields]
        gs_params = itertools.product(*gs_vals)
        for param_setting in gs_params:
            new_config = self.copy_config()
            for param,setting in zip(self.grid_search_fields,param_setting):
                new_config.__dict__[param] = setting
            yield new_config
