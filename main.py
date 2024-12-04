import pickle
from load import load_data, load_testing_data
from utils import create_dictionaries, create_transition_matrix, create_emission_matrix, compute_accuracy
from viterbi import initialize, forward_pass, backward_pass

def prepare_model(training_file, alpha = 0.001):
    training_corpus, vocab = load_data(training_file)

    with open('./model_weights/vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file)
    print('Vocab created and stored')

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

    with open('./model_weights/states.pkl', 'wb') as file:
        pickle.dump(states, file)
    print('States created and stored')

    
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



def inference(testing_file):

    testing_corpus, states, A, B, tag_counts, test_words, vocab = load_testing_data(testing_file)

    ################## Viterbi Algo ##################

    best_probs, best_paths = initialize(states, A, B, tag_counts, test_words, vocab)
    print('Viterbi Initialization Complete')

    best_probs, best_paths = forward_pass(A, B, test_words, vocab, best_probs, best_paths)
    print('Viterbi Forward Pass Complete')

    pred = backward_pass(states, best_probs, best_paths)
    print('Viterbi Backward Pass Complete')

    accuracy = compute_accuracy(testing_corpus, pred)
    print('Accuracy:', accuracy)

    return accuracy

inference('data/WSJ_test.pos')