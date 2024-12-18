import numpy as np
from collections import defaultdict
import string 
punct = set(string.punctuation)

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]
unknown_tokens = ["--unk_digit--", "--unk_punct--", "--unk_upper--", "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--", "--unk--"]

def assign_unk(word):
    # Digits
    if any(char.isdigit() for char in word):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in word):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in word):
        return "--unk_upper--"

    # Nouns
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"


def get_word_tag_split(line, vocab = None):
    if not line.split(): return '--n--','--s--'
    else:
        word, tag = line.split()
        if vocab and word not in vocab: word = assign_unk(word)
        return word, tag

def extract_words(corpus, vocab = None):
    words = []
    for line in corpus:
        word, tag = get_word_tag_split(line, vocab)
        words.append(word)
    return words
    
def create_vocab_from_list(voc_l):
    vocab = {}
    for i, word in enumerate(sorted(voc_l)): vocab[word] = i
    return vocab

def create_vocab_from_corpus(corpus):
    words = extract_words(corpus)
    freq_table = {}
    for word in words:
        freq_table[word] = freq_table.get(word,0)+1

    voc_l = []
    for key, value in freq_table.items():
        if value>1: voc_l.append(key)

    for unk in unknown_tokens:
        voc_l.append(unk)
    
    vocab = create_vocab_from_list(voc_l)
    return vocab

def create_dictionaries(training_corpus, vocab):
    tag_counts, transition_counts, emission_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    i = 0
    prev_tag = '--s--'
    for line in training_corpus:
        i += 1

        word, tag = get_word_tag_split(line, vocab)

        tag_counts[tag] += 1
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1

        prev_tag = tag
    
    return tag_counts, transition_counts, emission_counts


def create_transition_matrix(tag_counts, transition_counts, alpha):
    all_tags = sorted(tag_counts.keys())
    tot_tags = len(all_tags)

    transition_matrix = np.zeros((tot_tags, tot_tags))

    for i,prev_tag in enumerate(all_tags):
        for j,curr_tag in enumerate(all_tags):
            transition_matrix[i,j] =  (transition_counts.get((prev_tag, curr_tag),0)+alpha)/(tag_counts.get(prev_tag,0)+alpha*tot_tags)
    
    return transition_matrix


def create_emission_matrix(tag_counts, emission_counts, alpha, vocab):
    all_tags = sorted(tag_counts.keys())
    tot_tags = len(all_tags)

    vocab = list(vocab)
    tot_words = len(vocab)

    emission_matrix = np.zeros((tot_tags, tot_words))

    for i,tag in enumerate(all_tags):
        for j, word in enumerate(vocab):
            emission_matrix[i,j] =  (emission_counts.get((tag, word),0)+alpha)/(tag_counts.get(tag,0)+alpha*tot_words)
    
    return emission_matrix

def compute_accuracy(y, pred):
    correct, tot = 0, 0
    for prediction, y in zip(pred, y):
        word_tag_tuple = y.split()
        if len(word_tag_tuple) != 2: continue
        word, tag = word_tag_tuple

        if prediction == tag: correct+=1
        tot += 1
    
    return correct/tot
