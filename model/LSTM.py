import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import pymongo
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json


class LSTM_Classifier:
    def __init__(self, data=None, train=False, max_words=3000, labels=['class_1', 'class_2']):
        '''
        It receives data as pandas to numpy array of X trianing data and Y classes
        '''
        self.data = data
        self.corpus = None
        self.model = None
        self.max_words = max_words
        self.labels = labels
        self.tokenizer = None

        if train:
            self.__process_data__()
            self.__create_model__()
            self.__process__corpus__()
            self.__fit__()
        else:
            with open('corpus.json', 'r') as corpus_file:
                self.corpus = json.load(corpus_file)
            # self.model = keras.models.load_model('model.h5')
            self.tokenizer = Tokenizer(num_words=self.max_words)
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            # and create a model from that
            self.model = model_from_json(loaded_model_json)
            # and weight your nodes with your saved values
            self.model.load_weights('model.h5')

    def convert_text_to_index_array(self, text):
        words = kpt.text_to_word_sequence(text)
        wordIndices = []
        for word in words:
            if word in self.corpus:
                wordIndices.append(self.corpus[word])
            else:
                print("'%s' not in training corpus; ignoring." %(word))
        return wordIndices
    
    def __process_data__(self, ):
        data = self.data
        # train data
        self.train_x = [x[0] for x in data]
        self.train_y = np.asarray([x[1] for x in data])
        # print(self.train_x, self.train_y)


    def __process__corpus__(self, ):
        # fitting
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(self.train_x)

        # Tokenizers come with a convenient list of words and IDs
        dictionary = self.tokenizer.word_index
        # Let's save this out so we can use it later
        with open('corpus.json', 'w') as dictionary_file:
            json.dump(dictionary, dictionary_file)


        def convert_text_to_index_array(text):
            # one really important thing that `text_to_word_sequence` does
            # is make all texts the same length -- in this case, the length
            # of the longest text in the set.
            return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

        allWordIndices = []
        # for each tweet, change each token to its ID in the Tokenizer's word_index
        for text in self.train_x:
            wordIndices = convert_text_to_index_array(text)
            allWordIndices.append(wordIndices)

        # now we have a list of all tweets converted to index arrays.
        # cast as an array for future usage.
        allWordIndices = np.asarray(allWordIndices)

        # create one-hot matrices out of the indexed tweets
        self.train_x = self.tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
        # treat the labels as categories
        self.train_y = keras.utils.to_categorical(self.train_y, 2)

    def __create_model__(self, ):
        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(self.max_words,), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
    
    def __fit__(self, ):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.train_x, self.train_y, batch_size=32, epochs=5, verbose=1, validation_split=0.1, shuffle=True)

        model_json = self.model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model_json)

        self.model.save_weights('model.h5')
    
    def predict(self, text):

        # format your input for the neural net
        testArr = self.convert_text_to_index_array(text)
        input = self.tokenizer.sequences_to_matrix([testArr], mode='binary')
        # predict which bucket your input belongs in
        pred = self.model.predict(input)
        print(text)
        self.labels = self.labels[::-1]
        # and print it for the humons, pred[0][np.argmax(pred)] * 100
        prediction = self.labels[np.argmax(pred)]
        print("the show is: %s" % (prediction))
        return prediction
