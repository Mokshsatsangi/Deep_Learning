import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()

def create_lexicon(file_1, file_2):
    lexicon = []
    with open(file_1) as f1:
        content_f1 = f1.readlines()
        for i in content_f1:
            positive = word_tokenize(i)
            lexicon += list(positive)

    with open(file_2) as f2:
        content_f2 = f2.readlines()
        for j in content_f2:
            negative = word_tokenize(j)
            lexicon += list(negative)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    word_count = Counter(lexicon)
    lex = []
    for w in word_count:
        if 2000 > word_count[w] > 14:
            lex.append(w)
    print('\nlength of lexicon is ' + str(len(lex)))
    #print(lex)
    return lex

#lex = create_lexicon('D:/user/Documents/Python codes/Datasets/pos.txt', 'D:/user/Documents/Python codes/Datasets/neg.txt')
#np.save('D:/user/Documents/Python codes/ANN_codes/Sentiment_lexicon.npy', lex)
def sample_handling(file, lexicon, classification):
    feature_set = []

    with open(file) as f:
        contents = f.readlines()
        for i in contents:
            current_words = word_tokenize(i)
            current_words = [lemmatizer.lemmatize(j) for j in current_words]
            features = np.zeros(len(lexicon))
            for w in current_words:
                if w in lexicon:
                    id = lexicon.index(w)
                    features[id] += 1
            features = list(features)
            feature_set.append([features, classification])
    print('\nLength of feature_set is ' + str(len(feature_set)))
    #print(feature_set)
    return feature_set

#lex =  create_lexicon('D:/user/Documents/Python codes/Datasets/pos.txt', 'D:/user/Documents/Python codes/Datasets/neg.txt')
#sample_handling('D:/user/Documents/Python codes/Datasets/pos.txt', lex, [1, 0])
def assemble(positive, negative, test_size=0.1):
    lex =  create_lexicon(positive, negative)
    features = []
    features += sample_handling('D:/user/Documents/Python codes/Datasets/pos.txt', lex, [1, 0])
    features += sample_handling('D:/user/Documents/Python codes/Datasets/neg.txt', lex, [0, 1])
    random.shuffle(features)

    features = np.array(features)
    print('\nlength of features is ' + str(len(features)))
    testing_size = int(test_size * len(features))
    print('\ntesting size is ' + str(testing_size))

    train_x = list(features[:,0][:-testing_size])
    print('\nSize of train_x is ' + str(len(train_x)))
    train_y = list(features[:,1][:-testing_size])
    print('\nSize of train_y is ' + str(len(train_y)))

    test_x = list(features[:,0][-testing_size:])
    print('\nSize of test_x is ' + str(len(test_x)))
    test_y = list(features[:,1][-testing_size:])
    print('\nSize of test_y is ' + str(len(test_y)))

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = assemble('D:/user/Documents/Python codes/Datasets/pos.txt', 'D:/user/Documents/Python codes/Datasets/neg.txt')
    with open('Positive-Negative.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
