# Sentiment Analysis using Keras
# Data Source: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
# Reference: https://realpython.com/python-keras-text-classification/#your-first-keras-model

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt

filepath_dict = {'yelp':   'sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb':   'sentiment labelled sentences/imdb_labelled.txt'}
review_list = []
for key, value in filepath_dict.items():
    df = pd.read_csv(value, names=['sentence', 'label'], sep='\t')
    df['source'] = key
    review_list.append(df)

reviews_train = pd.concat(review_list)

for source in reviews_train['source'].unique():
    X = reviews_train['sentence']
    Y = reviews_train['label']
    # train test set split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size= 0.3,
                                                        random_state=0)
# Word Embedding

    tokenizer = Tokenizer(num_words=5000)
# Updates internal vocabulary based on a list of texts
    tokenizer.fit_on_texts(X_train)
# Transforms each text in texts to a sequence of integers
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    # Building embedding layer
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    # dense vector size
    output_dim = 50
    maxlen = 100
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
    sequential = Sequential()
    sequential.add(layers.Embedding(input_dim=vocab_size, output_dim=output_dim, input_length= maxlen))
    sequential.add(layers.GlobalMaxPool1D())
    sequential.add(layers.Dense(10, activation='relu'))
    sequential.add(layers.Dense(1, activation='sigmoid'))
    sequential.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(sequential.summary())
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



