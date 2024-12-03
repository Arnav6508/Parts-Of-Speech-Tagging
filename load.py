def load_data():
    with open("./data/WSJ_train.pos", 'r') as f:
        training_corpus = f.readlines()
    with open("./data/WSJ_test.pos", 'r') as f:
        testing_corpus = f.readlines()

    with open("./data/hmm_vocab.txt", 'r') as f:
        voc_l = f.read().split('\n')

    return training_corpus, testing_corpus, voc_l