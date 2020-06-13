import numpy as np
import time
import string
import torch
import os


from main.objects.Batcher import Batcher
from main.objects.Writer import Writer
from main.objects.Scorer import Scorer

'''
Evaluator object which scores the model and write the model predictions 
'''
class Evaluator(object):
    def __init__(self, config, input_type, exp_dir, list_k, labeled_file=None, output_file=None, is_write=False):
        '''
        param config: configuration to use for evaluation 
        param vocab: vocabulary to use 
        param tokenizer: tokenizer to use 
        param input_type: input type of either dev/test
        param exp_dir: experiment directory to save output
        param list_k: list of k to evaluate hits@k
        param labeled_file: labeled file to use for labels (default is specified in config )
        param output_file: output file to use for writing prediction 
        '''
        self.batcher = Batcher(config, input_type, labeled_file)
        self.config = config
        self.input_type = input_type
        self.list_k = list_k
        self.is_write = is_write
        
        if self.input_type == "dev":
            self.best_dev_score = 0
            self.score_filename = os.path.join(exp_dir, "dev_scores.json")
            self.best_model_filename = os.path.join(exp_dir, "best_model")

            if is_write:
                if output_file is None:
                    self.dev_file = os.path.join(exp_dir, "dev.prediction")
                else:
                    self.dev_file = output_file
                self.writer = Writer(self.dev_file)

        elif self.input_type == "test":
            if output_file is not None:
                self.test_file = output_file
            else:
                self.test_file = os.path.join(exp_dir, "test.predictions")
            self.score_filename = os.path.join(exp_dir, "test_scores.json")
            self.writer = Writer(self.test_file)

        self.output_file = None
        if output_file:
            self.output_file = output_file

        self.score = True
        if self.output_file is not None and "shard" in self.output_file:
            self.score = False


    def evaluate(self, model, train_num_batches, train_loss):
        '''
        Evaluates the model by scoring it and writing its predictions 
        
        param train_num_batches: number of batches the model has trained on 
        param train_loss: loss for current batch when training 
        '''
        if self.score is True:
            scorer = Scorer(self.list_k, self.score_filename, train_num_batches, train_loss)

        for batch_str, batch_ids, lbls, end_block in self.batcher.get_dev_test_batches():
            scores = model.score_dev_test_batch(batch_ids)

            if isinstance(scores, np.ndarray):
                scores = list(scores)
            else:
                scores = list(scores.cpu().data.numpy().squeeze(1))

            if self.score == True:
                scorer.add_batch_pred_scores(scores, lbls, end_block)

            if self.input_type == "test" or self.is_write:
                self.writer.add_batch_pred_lab(batch_str, lbls, scores)

        if self.score == True:
            map_score = scorer.calc_scores()

            # Calculate the scores and save if best so far
            if self.input_type == "dev":
                if map_score > self.best_dev_score:
                    torch.save(model, self.best_model_filename)
                    self.best_dev_score = map_score
                    self.best_score_iter = 0
                else:
                    self.best_score_iter += 1


