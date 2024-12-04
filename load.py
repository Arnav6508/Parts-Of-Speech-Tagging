import pickle
from utils import create_vocab, extract_words

def load_data(training_data_file):
    with open(training_data_file, 'r') as f:
        training_corpus = f.readlines()

    with open('data/hmm_vocab.txt', 'r') as f:
        voc_l = f.read().split('\n')
    vocab = create_vocab(voc_l)

    return training_corpus, vocab


def load_testing_data(testing_file):
    with open(testing_file, 'r') as f:
        testing_corpus = f.readlines()
    
    with open("./model_weights/states.pkl", 'rb') as f:
        states = pickle.load(f)

    with open("./model_weights/dictionaries.pkl", 'rb') as f:
        dictionaries = pickle.load(f)
    
    with open("./model_weights/matrices.pkl", 'rb') as f:
        matrices = pickle.load(f)

    with open("./data/hmm_vocab.txt", 'r') as f:
        voc_l = f.read().split('\n')
    vocab = create_vocab(voc_l)
    
    test_words = extract_words(testing_corpus, vocab)

    return testing_corpus, states, matrices['transition_matrix'], matrices['emission_matrix'], dictionaries['tag_counts'], test_words, vocab