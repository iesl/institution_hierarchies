from sklearn.metrics import average_precision_score
from main.utils.util import save_dict_to_json

'''
Scorer object which scores the predictions of a model 
'''


class Scorer(object):
    def __init__(self, list_k, score_filename, train_num_batches, train_loss):
        '''
        param list_k: list of k to evaluate hits@k
        param score_filename: filename to write scores
        param train_num_batches: number of batches model has been trained on
        '''
        self.avg_precs = []
        self.list_k = list_k
        self.dict_avg_hits_at_k = {}

        self.scores_obj = {}

        if train_num_batches != -1:
            self.scores_obj["train_num_batches"] = train_num_batches
        if train_loss != -1:
            self.scores_obj["train_loss"] = train_loss

        self.score_filename = score_filename
        self.queries = []
        self.scores = []
        self.labels = []
        for k in list_k:
            self.dict_avg_hits_at_k[k] = []

    def add_batch_pred_scores(self, scores, labels, end_block):
        '''
        Adds batch of predicted scores and calculates scores if end of current block of query

        param qry_tk: query_tokens for current batch
        param scores: scores predicted for current batch
        param labels: labels for current batch
        param end_block: whether the current batch is the end of a current block fo query or not
        '''
        self.scores.extend(scores)
        self.labels.extend(labels)

        if end_block:
            # Ensures there is at least 1 true positive
            if sum(self.labels) > 0:
                self.avg_precs.append(average_precision_score(self.labels, self.scores))

                zipped = zip(self.scores, self.labels)
                sort_zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
                sorted_labels = list(zip(*sort_zipped))[1]

                for k in self.list_k:
                    top_k_labels = sorted_labels[:k]
                    hits_at_k = sum(top_k_labels) / min(k, sum(self.labels))
                    self.dict_avg_hits_at_k[k].append(hits_at_k)

            self.scores = []
            self.labels = []

    def load_blocks(self):
        '''
        Loads and stores all the blocks

        return: result_labels
        return: result_scores
        '''
        result_labels = []
        result_scores = []
        current_qry = ""
        current_block_scores = []
        current_block_labels = []

        for (qry, label, score) in zip(self.queries, self.labels, self.scores):
            if qry != current_qry and current_qry != "":
                result_labels.append(current_block_labels)
                result_scores.append(current_block_scores)
                current_block_labels = []
                current_block_scores = []
            current_block_labels.append(int(label))
            current_block_scores.append(float(score))
            current_qry = qry
        result_labels.append(current_block_labels)
        result_scores.append(current_block_scores)
        return result_labels, result_scores

    def calc_scores(self, is_write=True):
        '''
        Calculates all the scores that haven been added by the batch

        return: map score for calculating dev score
        '''

        # result_labels, result_scores = self.load_blocks()
        # map_score = float(eval_map(result_labels,result_scores))
        map_score = float(sum(self.avg_precs) / len(self.avg_precs))
        self.scores_obj["map"] = map_score

        for (k, avg_hits_at_k) in self.dict_avg_hits_at_k.items():
            hits_at_k_label = "hits_at_" + str(k)
            hits_at_k_score = float(sum(avg_hits_at_k) / len(avg_hits_at_k))
            self.scores_obj[hits_at_k_label] = hits_at_k_score

        for k, v in self.scores_obj.items():
            self.scores_obj[k] = float(str("%.3f") % v)

        if is_write:
            save_dict_to_json(self.scores_obj, self.score_filename)

        return map_score

