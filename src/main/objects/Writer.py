
'''
Writer object to write the scores
'''
class Writer(object):
    def __init__(self, test_file):
        '''
        param test_file: test file to write the scores to
        '''
        self.test_file = open(test_file, 'w+')

    def add_batch_pred_lab(self, batch_str, lbls, scores):
        '''
        Writes batch of prediction

        param batch_str: batch of (qry, cnd) strings
        param lbl: batch of labels
        param score: batch of scores
        '''
        for (ind_str, lbl, score) in zip(batch_str, lbls, scores):
            tab_splits = [ind_str[0], ind_str[1], str(lbl), ("{:.3f}".format(float(score)))]
            line = '\t'.join(tab_splits) + '\n'
            self.test_file.write(line)