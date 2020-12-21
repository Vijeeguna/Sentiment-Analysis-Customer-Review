# Sentiment Analysis using Keras
# Data Source: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
# Reference: https://realpython.com/python-keras-text-classification/#your-first-keras-model
# GloVe Word Embedding Source: http://nlp.stanford.edu/data/glove.6B.zip

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

filepath_dict = {'yelp': 'sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb': 'sentiment labelled sentences/imdb_labelled.txt'}
review_list = []
for key, value in filepath_dict.items():
    df = pd.read_csv(value, names=['sentence', 'label'], sep='\t')
    df['source'] = key
    review_list.append(df)

reviews_train = pd.concat(review_list)


# Create CNN Model
def create_CNN_Model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, optimizer):
    sequential = Sequential()
    sequential.add(layers.Embedding(vocab_size, embedding_dim,
                                    input_length=maxlen))
    sequential.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    sequential.add(layers.GlobalMaxPool1D())
    sequential.add(layers.Dense(10, activation='relu'))
    sequential.add(layers.Dense(1, activation='sigmoid'))
    sequential.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    return sequential


for source in reviews_train['source'].unique():
    X = reviews_train['sentence']
    Y = reviews_train['label']
    # train test set split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=0)
    embedding_dim = 50
    maxlen = 100
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[5000],
                      embedding_dim=[50],
                      maxlen=[100],
                      optimizer= ['adam', 'rmsprop'])
    sequential = KerasClassifier(build_fn=create_CNN_Model,
                                 epochs= 20,
                                 batch_size=10,
                                 verbose=False)
    grid = RandomizedSearchCV(estimator= sequential,
                              param_distributions=param_grid,
                              cv = 10,
                              n_iter=5)
    result = grid.fit(X_train, Y_train)
    score = grid.score(X_test, Y_test)
    print('Optimal parameter values are: ', result.best_estimator_)
    print('Accuracy of the fit is: ', score)
    # 82% Accuracy


