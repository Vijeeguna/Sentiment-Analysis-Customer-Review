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


# Creating pretrained embedding space
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


for source in reviews_train['source'].unique():
    X = reviews_train['sentence']
    Y = reviews_train['label']
    # train test set split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=0)
    # Using Pretrained Word Embeddings
    # Using GloVe Word Embedding
    embedding_dim = 50
    maxlen = 100
    num_words = 5000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
    embedding_matrix = create_embedding_matrix('glove.6B/glove.6B.50d.txt',
                                               tokenizer.word_index, embedding_dim)
    sequential = Sequential()
    sequential.add(layers.Embedding(vocab_size, embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=True))
    sequential.add(layers.Conv1D(128, 5, activation='relu'))
    sequential.add(layers.GlobalMaxPool1D())
    sequential.add(layers.Dense(10, activation='relu'))
    sequential.add(layers.Dense(1, activation='sigmoid'))
    sequential.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    history = sequential.fit(X_train, Y_train,
                             epochs=20,
                             verbose=False,
                             validation_data=(X_test, Y_test),
                             batch_size=10)
    print('Accuracy of the simple neural network model for ', source, ' reviews is :', history.history['accuracy'])
    # testing accuracy
    print('Accuracy of the simple neural network model for ', source, ' reviews is :', history.history['val_accuracy'])
    # plot performance
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'b', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()




