import numpy as np

def get_set_difference(btc_qry_tk, btc_cnd_tk):
    btc_qry_min_cnd_tk = []
    btc_cnd_min_qry_tk = []

    for (qry_tk, cnd_tk) in zip(btc_qry_tk, btc_cnd_tk):
        btc_qry_min_cnd_tk.append(set(qry_tk).difference(set(cnd_tk)))
        btc_cnd_min_qry_tk.append(set(cnd_tk).difference(set(qry_tk)))

    return np.asarray(btc_qry_min_cnd_tk), np.asarray(btc_cnd_min_qry_tk)

def get_set_intersect(btc_qry_tk, btc_cnd_tk):
    batch_qry_insct_cnd_tk = []

    for (qry_tk, cnd_tk) in zip(btc_qry_tk, btc_cnd_tk):
        batch_qry_insct_cnd_tk.append(set(qry_tk).intersection(set(cnd_tk)))

    return np.asarray(batch_qry_insct_cnd_tk)

def get_set_union(btc_qry_tk, btc_cnd_tk):
    btc_qry_unn_cnd_tk = []

    for (qry_tk, cnd_tk) in zip(btc_qry_tk, btc_cnd_tk):
        btc_qry_unn_cnd_tk.append(set(qry_tk).union(set(cnd_tk)))

    return np.asarray(btc_qry_unn_cnd_tk)

def lookup_qry_cnd(vocab, btc_qry_tk, btc_cnd_tk):
    '''
    Returns the list of token indices given list of tokens passed in 
    
    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_cnd_tk: batch of candidate tokens 
    return: batch of query token indices 
    return: batch of candidate token indices 
    '''
    btc_qry_lkup = []
    btc_cnd_lkup = []

    for qry_tk, cnd_tk in zip(btc_qry_tk, btc_cnd_tk):
        btc_qry_lkup.append(vocab.to_ints(qry_tk))
        btc_cnd_lkup.append(vocab.to_ints(cnd_tk))

    return np.asarray(btc_qry_lkup), np.asarray(btc_cnd_lkup)

def lookup_qry_pos_neg(vocab, btc_qry_tk, btc_pos_tk, btc_neg_tk):
    '''
    Returns the list of token indices given list of tokens passed in 
    
    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_pos_tk: batch of positive tokens 
    param btc_neg_tk: batch of negative tokens 
    return: batch of query token indices 
    return: batch of positive token indices
    return: batch of negative token indices  
    '''
    btc_qry_lkup = []
    btc_pos_lkup = []
    btc_neg_lkup = []

    for qry_tk, pos_tk, neg_tk in zip(btc_qry_tk, btc_pos_tk, btc_neg_tk):
        btc_qry_lkup.append(vocab.to_ints(qry_tk))
        btc_pos_lkup.append(vocab.to_ints(pos_tk))
        btc_neg_lkup.append(vocab.to_ints(neg_tk))

    return np.asarray(btc_qry_lkup), np.asarray(btc_pos_lkup), np.asarray(btc_neg_lkup)

def lookup_qry_pos_neg_diff_insct(vocab, btc_qry_min_pos_tk, btc_pos_min_qry_tk, btc_qry_insct_pos_tk, btc_qry_min_neg_tk, btc_neg_min_qry_tk, btc_qry_insct_neg_tk):
    '''
    Returns the list of token indices given list of tokens passed in 
    
    param vocab: vocab object
    param btc_qry_min_pos_tk: batch of query set difference positive tokens 
    param btc_pos_min_qry_tk: batch of positive set difference query tokens 
    param btc_qry_insct_pos_tk: batch of query set intersect positive tokens 
    param btc_qry_min_neg_tk: batch of query set difference negative tokens 
    param btc_neg_min_qry_tk: batch of negative set difference query tokens 
    param btc_qry_insct_neg_tk: batch of query set intersect negative  tokens 
    return: batch of query set difference positive indices 
    return: batch of positive set difference query indices 
    return: batch of query set intersect positive indices 
    return: batch of query set difference negative indices 
    return: batch of negative set difference query indices 
    return: batch of query set intersect negative  indices 
    '''
    btc_qry_min_pos_lkup = []
    btc_pos_min_qry_lkup = []
    btc_qry_insct_pos_lkup = []
    btc_qry_min_neg_lkup = []
    btc_neg_min_qry_lkup = []
    btc_qry_insct_neg_lkup = []

    for qry_min_pos_tk, pos_min_qry_tk, qry_insct_pos_tk, qry_min_neg_tk, neg_min_qry_tk, qry_insct_neg_tk in \
        zip(btc_qry_min_pos_tk, btc_pos_min_qry_tk, btc_qry_insct_pos_tk, btc_qry_min_neg_tk, btc_neg_min_qry_tk, btc_qry_insct_pos_tk):
        btc_qry_min_pos_lkup.append(vocab.to_ints(qry_min_pos_tk))
        btc_pos_min_qry_lkup.append(vocab.to_ints(pos_min_qry_tk))
        btc_qry_insct_pos_lkup.append(vocab.to_ints(qry_insct_pos_tk))
        btc_qry_min_neg_lkup.append(vocab.to_ints(qry_min_neg_tk))
        btc_neg_min_qry_lkup.append(vocab.to_ints(neg_min_qry_tk))
        btc_qry_insct_neg_lkup.append(vocab.to_ints(qry_insct_neg_tk))

    return np.asarray(btc_qry_min_pos_lkup), np.asarray(btc_pos_min_qry_lkup), np.asarray(btc_qry_insct_pos_lkup), \
            np.asarray(btc_qry_min_neg_lkup), np.asarray(btc_neg_min_qry_lkup), np.asarray(btc_qry_insct_neg_lkup)


def lookup_qry_cnd_diff_insct(vocab, btc_qry_min_cnd_tk, btc_cnd_min_qry_tk, btc_qry_insct_cnd_tk):
    '''
    Returns the list of token indices given list of tokens passed in 
    
    param vocab: vocab object
    param btc_qry_min_pos_tk: batch of query set difference candidate tokens 
    param btc_pos_min_qry_tk: batch of candidate set difference query tokens 
    param btc_qry_insct_pos_tk: batch of query set intersect candidate tokens 
    return: batch of query set difference candidate indices 
    return: batch of candidate set difference query indices 
    return: batch of query set intersect candidate indices 
    '''
    btc_qry_min_cnd_lkup = []
    btc_cnd_min_qry_lkup = []
    btc_qry_insct_cnd_lkup = []

    for qry_min_cnd_tk, cnd_min_qry_tk, qry_insct_cnd_tk in \
            zip(btc_qry_min_cnd_tk, btc_cnd_min_qry_tk, btc_qry_insct_cnd_tk):
        btc_qry_min_cnd_lkup.append(vocab.to_ints(qry_min_cnd_tk))
        btc_cnd_min_qry_lkup.append(vocab.to_ints(cnd_min_qry_tk))
        btc_qry_insct_cnd_lkup.append(vocab.to_ints(qry_insct_cnd_tk)) 

    return np.asarray(btc_qry_min_cnd_lkup), np.asarray(btc_cnd_min_qry_lkup), np.asarray(btc_qry_insct_cnd_lkup)


def get_qry_pos_neg_tok_lookup(vocab, qry_tk, pos_tk, neg_tk):
    '''
    Returns the list of token indices given list of tokens passed in

    param vocab: vocab object
    param btc_qry_tk: batch of query tokens
    param btc_pos_tk: batch of positive tokens
    param btc_neg_tk: batch of negative tokens
    return: batch of query token indices
    return: batch of positive token indices
    return: batch of negative token indices
    return: batch of qry minus pos tokens indices
    return: batch of pos minus qry tokens indices
    return: batch of qry intersect pos tokens indices
    return: batch of qry minus neg tokens indices
    return: batch of neg minus qry tokens indices
    return: batch of qry intersect neg tokens  indices

    '''
    qry_lkup, pos_lkup, neg_lkup = lookup_qry_pos_neg(vocab, qry_tk, pos_tk, neg_tk)
    return qry_lkup, pos_lkup, neg_lkup


def get_qry_pos_neg_insct_min_tok_lookup(vocab, qry_tk, pos_tk, neg_tk):
    '''
    Returns the list of token indices given list of tokens passed in 
    
    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_pos_tk: batch of positive tokens 
    param btc_neg_tk: batch of negative tokens 
    return: batch of query token indices 
    return: batch of positive token indices
    return: batch of negative token indices
    return: batch of qry minus pos tokens indices
    return: batch of pos minus qry tokens indices
    return: batch of qry intersect pos tokens indices
    return: batch of qry minus neg tokens indices
    return: batch of neg minus qry tokens indices
    return: batch of qry intersect neg tokens  indices

    '''
    qry_min_pos_tk, pos_min_qry_tk = get_set_difference(qry_tk, pos_tk)
    qry_insct_pos_tk = get_set_intersect(qry_tk, pos_tk)
    qry_min_neg_tk, neg_min_qry_tk = get_set_difference(qry_tk, neg_tk)
    qry_insct_neg_tk = get_set_intersect(qry_tk, neg_tk)

    qry_lkup, pos_lkup, neg_lkup = lookup_qry_pos_neg(vocab, qry_tk, pos_tk, neg_tk)

    qry_min_pos_lkup, pos_min_qry_lkup, qry_insct_pos_lkup, qry_min_neg_lkup, neg_min_qry_lkup, qry_insct_neg_lkup = lookup_qry_pos_neg_diff_insct(vocab, qry_min_pos_tk, pos_min_qry_tk, qry_insct_pos_tk, qry_min_neg_tk, neg_min_qry_tk, qry_insct_neg_tk)

    return qry_lkup, pos_lkup, neg_lkup, qry_min_pos_lkup, pos_min_qry_lkup, qry_insct_pos_lkup, qry_min_neg_lkup, neg_min_qry_lkup, qry_insct_neg_lkup


def get_qry_cnd_tok_lookup(vocab, qry_tk, cnd_tk):
    '''
    Returns the list of token indices given list of tokens passed in

    param vocab: vocab object
    param btc_qry_tk: batch of query tokens
    param btc_cnd_tk: batch of candidate tokens
    return: batch of query token indices
    return: batch of candidate token indices
    return: batch of qry min cnd token indices
    return: batch of cnd min qry token indices
    return: batch of qry intersect cnd token indices
    '''
    qry_lkup, cnd_lkup = lookup_qry_cnd(vocab, qry_tk, cnd_tk)
    return qry_lkup, cnd_lkup

def get_qry_cnd_insct_min_tok_lookup(vocab, qry_tk, cnd_tk):
    '''
    Returns the list of token indices given list of tokens passed in 
    
    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_cnd_tk: batch of candidate tokens 
    return: batch of query token indices 
    return: batch of candidate token indices
    return: batch of qry min cnd token indices
    return: batch of cnd min qry token indices
    return: batch of qry intersect cnd token indices
    '''
    qry_min_cnd_tk, cnd_min_qry_tk = get_set_difference(qry_tk, cnd_tk)
    qry_insct_cnd_tk = get_set_intersect(qry_tk, cnd_tk)

    qry_lkup, cnd_lkup = lookup_qry_cnd(vocab, qry_tk, cnd_tk)
    qry_min_cnd_lkup, cnd_min_qry_lkup, qry_insct_cnd_lkup = lookup_qry_cnd_diff_insct(vocab, qry_min_cnd_tk, cnd_min_qry_tk, qry_insct_cnd_tk)

    return qry_lkup, cnd_lkup, qry_min_cnd_lkup, cnd_min_qry_lkup, qry_insct_cnd_lkup

def get_qry_cnd_tok(qry_tk, cnd_tk):
    qry_min_cnd_tk, cnd_min_qry_tk = get_set_difference(qry_tk, cnd_tk)
    qry_insct_cnd_tk = get_set_intersect(qry_tk, cnd_tk)

    return qry_tk, cnd_tk, qry_min_cnd_tk, cnd_min_qry_tk, qry_insct_cnd_tk