# Sentiment Analysis using Keras
# Data Source: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
# Reference: https://realpython.com/python-keras-text-classification/#your-first-keras-model

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
# Bag of Words
# fit learns a vocabulary dictionary of all tokens in the raw document
# transform learns this vocab dict and returns a document term matrix
    vectorizer = CountVectorizer(lowercase= True, stop_words='english')
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()
# building the baseline model
# Logistic regression
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print('Accuracy for', source, ' reviews is :', score)
# Simple neural network using Keras
    sequential = Sequential()
    sequential.add(layers.Dense(10, input_dim= X_train.shape[1], activation='relu'))
    sequential.add(layers.Dense(1, activation='sigmoid'))
# use binary_crossentropy loss since this is a binary classification model
    sequential.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    print(sequential.summary())
    history = sequential.fit(X_train, Y_train,
                             verbose= False,
                             batch_size=10,
                             epochs= 10,
                             validation_data=(X_test, Y_test))
    print('Accuracy of the simple neural network model for ', source,' reviews is :', history.history['accuracy'])
# testing accuracy
    print('Accuracy of the simple neural network model for ', source, ' reviews is :', history.history['val_accuracy'])
# plot performance
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'b', label= 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'b', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Model is over-fit since the training set performs well with 96% accuracy but this drops to 78% in validation set :/





