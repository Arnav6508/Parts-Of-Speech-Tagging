import math
import numpy as np

def initialize(states, A, B, tag_counts, corpus, vocab):
    tot_tags = len(tag_counts)
    tot_words = len(corpus)
    s_idx = states.index('--s--')

    best_probs = np.zeros((tot_tags, tot_words))
    best_paths = np.zeros((tot_tags, tot_words), dtype=int)

    for i in range(tot_tags):
        best_probs[i,0] = math.log(A[s_idx,i]) + math.log(B[i,vocab[corpus[0]]])
        # print(A[s_idx,i],B[i,vocab[corpus[0]]])

    return best_probs, best_paths

def forward_pass(A, B, corpus, vocab, best_probs, best_paths):
    tot_tags = best_probs.shape[0]
    tot_words = len(corpus)

    for i in range(1,tot_words):
        # j = curr word tag
        for j in range(tot_tags):

            best_prob_j_i = float('-inf')
            best_path_j_i = None

            # k = prev word tag
            for k in range(tot_tags):

                best_prob_temp = best_probs[k,i-1] + math.log(A[k,j]) + math.log(B[j,vocab[corpus[i]]])
                if best_prob_temp>best_prob_j_i:
                    best_prob_j_i = best_prob_temp
                    best_path_j_i = k
                
            # best_probs stores the curr (word-tag) combination prob
            # best_paths stores the prev word ka tag
            best_probs[j,i] = best_prob_j_i
            best_paths[j,i] = best_path_j_i

    return best_probs, best_paths

def backward_pass(states, best_probs, best_paths):
    tot_tags = best_probs.shape[0]
    tot_words = best_paths.shape[1]

    # z stores the index of tag for curr word
    # pred stores the tag string for curr word
    z, pred = [None]*tot_words, [None]*tot_words
    best_prob_last_word = float('-inf')

    # Calc z, pred for last word in corpus
    for i in range(tot_tags):
        if best_prob_last_word<best_probs[i,-1]:
            best_prob_last_word = best_probs[i,-1]
            z[-1] = i
    pred[-1] = states[z[-1]]

    # Backtrack from the last word to first word
    for i in range(tot_words-1,0,-1):
        tag_i = z[i]
        z[i-1] = best_paths[tag_i,i]
        pred[i-1] = states[z[i-1]]
    
    return pred