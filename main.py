import pickle
import pandas as pd
from load import load_data
from utils import preprocess_test_data, create_vocab, create_dictionaries, create_transition_matrix, create_emission_matrix, compute_accuracy
from viterbi import initialize, forward_pass, backward_pass

def prepare_model(alpha = 0.001):
    training_corpus, testing_corpus, voc_l = load_data()
    vocab = create_vocab(voc_l)
    test_words = preprocess_test_data(vocab, 'data/test.words')

    ################## Dictionaries ##################

    tag_counts, transition_counts, emission_counts = create_dictionaries(training_corpus, vocab)
    states = sorted(list(tag_counts.keys()))

    dictionaries = {
        'tag_counts': tag_counts,
        'transition_counts': transition_counts,
        'emission_counts': emission_counts
    }
    with open('./model_weights/dictionaries.pkl', 'wb') as file:
        pickle.dump(dictionaries, file)

    print('Dictionaries created and stored')
    
    ################## Matrices ##################

    A = create_transition_matrix(tag_counts, transition_counts, alpha)
    B = create_emission_matrix(tag_counts, emission_counts, alpha, vocab)

    matrices = {
        'transition_matrix': A,
        'emission_matrix': B
    }

    with open(f'./model_weights/matrices.pkl', 'wb') as file:
        pickle.dump(matrices, file)

    print('Matrices created and stored')

    ################## Viterbi Algo ##################

    best_probs, best_paths = initialize(states, A, B, tag_counts, test_words, vocab)
    print('Viterbi Initialization Complete')

    best_probs, best_paths = forward_pass(A, B, test_words, vocab, best_probs, best_paths)
    print('Viterbi Forward Pass Complete')

    with open('./model_weights/best_probs.pkl','wb') as f:
        pickle.dump(best_probs, f)
    with open('./model_weights/best_paths.pkl','wb') as f:
        pickle.dump(best_paths, f)
    with open('./model_weights/states.pkl','wb') as f:
        pickle.dump(states, f)

    pred = backward_pass(states, best_probs, best_paths)
    print('Viterbi Backward Pass Complete')

    accuracy = compute_accuracy(testing_corpus, pred)
    print('Accuracy:', accuracy)

    return accuracy

