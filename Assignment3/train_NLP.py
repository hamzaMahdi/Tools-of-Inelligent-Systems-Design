# import required packages
import tensorflow as tf
from keras.utils import get_file
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

# for importing data
import os
import tarfile
import glob

# some text processing stuff 
import re
import string
import nltk


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# downloads data from original link and unzips it
def download_data():
  data_dir = get_file('aclImdb_v1.tar.gz', 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', cache_subdir = "datasets",
    hash_algorithm = "auto", extract = True, archive_format = "auto")
  my_tar = tarfile.open(data_dir)
  my_tar.extractall('./data/') # specify which folder to extract to
  my_tar.close()



# loads raw data from expected file path
# 0 is positive, 1 is negative
def load_data():
  # empty arrays
  train_pos = []
  test_pos = []
  train_neg = []
  test_neg = []
  # path to each set
  positive_train = glob.glob("data/aclImdb/train/pos/*.txt")
  positive_test = glob.glob("data/aclImdb/test/pos/*.txt")
  negative_train = glob.glob("data/aclImdb/train/neg/*.txt")
  negative_test = glob.glob("data/aclImdb/test/neg/*.txt")

  # load each set
  for file_path in positive_train:
      with open(file_path, 'r', encoding="utf8") as file_input:
          train_pos.append(file_input.read())
  for file_path in positive_test:
      with open(file_path, 'r', encoding="utf8") as file_input:
          test_pos.append(file_input.read())

  for file_path in negative_train:
      with open(file_path, 'r', encoding="utf8") as file_input:
          train_neg.append(file_input.read())

  for file_path in negative_test:
      with open(file_path, 'r', encoding="utf8") as file_input:
          test_neg.append(file_input.read())
  # label the sets
  train_labels = np.concatenate([np.zeros(len(train_pos)), np.ones(len(train_neg))])
  test_labels = np.concatenate([np.zeros(len(test_pos)), np.ones(len(test_neg))])
  train_data = train_pos
  train_data.extend(train_neg)

  test_data = test_pos
  test_data.extend(test_neg)

  return train_data, train_labels, test_data, test_labels



# text preprocessing helper functions
# the following resource was helpful in cleaning my dataset
# https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing

# remove punctuation
def remove_punctuation(input):
    return input.translate(str.maketrans('', '', string.punctuation))

# remove stop words
def remove_stopwords(input):
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(input).split() if word not in STOPWORDS])

# remove url
def remove_urls(input):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', input)

# remove HTML tags
def remove_html(input):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', input)


# load data and clean it up using some preprocessing functions
def load_and_preprocess():
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    # convert to lower case
    X_train = list(map(str.lower, X_train))
    X_test = list(map(str.lower, X_test))
    
    # remove punctuation
    X_train = list(map(remove_punctuation, X_train))
    X_test = list(map(remove_punctuation, X_test))
    
    # remove stop words
    # This link suggested we dont remove stop words, and gave a valid argument for it
    # Since I will be using some sort of reccurent network, I dont think stopwords will 
    # impact me a lot. I will be keeping them for now 
    # https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52
    # X_train = list(map(remove_stopwords, X_train))
    # X_test = list(map(remove_stopwords, X_test))
    
    # remove urls
    X_train = list(map(remove_urls, X_train))
    #X_test = list(map(remove_urls, X_test))
    
    # remove HTML stuff
    X_train = list(map(remove_html, X_train))
    X_test = list(map(remove_html, X_test))
    
    return X_train, y_train, X_test, y_test


# this function utilizes keras to tokenize text arrays into numpy arrays
def tokenize_text(vocab_size, X_train, X_test):
    tokenizer = Tokenizer(num_words=vocab_size)
    print('Tokenizing....')
    # tokenize text
    tokenizer.fit_on_texts(X_train) 
    X_train_encoded = tokenizer.texts_to_sequences(X_train)
    X_test_encoded = tokenizer.texts_to_sequences(X_test)
    
    return X_train_encoded, X_test_encoded


def pad_text(X_train_encoded, X_test_encoded, LARGEST_SENTENCE=300):
    X_train_ready = sequence.pad_sequences(X_train_encoded, maxlen=LARGEST_SENTENCE)
    X_test_ready = sequence.pad_sequences(X_test_encoded, maxlen=LARGEST_SENTENCE)
    
    return X_train_ready, X_test_ready

# plots loss and accuracy for training and validation
def plot_history(history, img_name):
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(accuracy)+1)
  plt.plot(epochs, accuracy, label='Training accuracy')
  plt.plot(epochs, val_accuracy, label='Validation accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend()
  plt.title(img_name)
  plt.savefig('accuracy_'+img_name+'.png')

  # loss history
  plt.figure()
  plt.plot(epochs, loss, label='Training loss')
  plt.plot(epochs, val_loss, label='Validation loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.title(img_name)
  plt.savefig('loss_'+img_name+'.png')
  plt.show()


# returns model and early stopping callback
def build_model(vocab_size, LARGEST_SENTENCE):
    # save the best model
    checkpoint_path="models/20607230_NLP_model.h5"
    keras_callbacks   = [
          EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.0001),
          ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    ]
    
    # see report for design decisions
    model = Sequential()
    model.add(Embedding(vocab_size, LARGEST_SENTENCE, input_length=LARGEST_SENTENCE))
    model.add(Conv1D(filters=32, kernel_size=3,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model, keras_callbacks


# ran into some memory issues so i had to do this
def gpu_settings():
    try:
        K.clear_session()
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

if __name__ == "__main__": 
    
    # fixes gpu memory issues (if any)
    # please comment if this gives you an issue as Ive used tf2
    # I tried to keep the code flexible so it doesnt fail with tf1
    gpu_settings()
    
	# 1. load your training data
    # uncomment below if you want to download the dataset
    # download_data()
    # load data
    X_train, y_train, X_test, y_test = load_and_preprocess()
    
    # tokenize text
    vocab_size = 10000 # There are 24904 words in training data but I want to use less to make training easier
    X_train_encoded, X_test_encoded = tokenize_text(vocab_size, X_train, X_test)
    
    # Originally I wanted to use a larger number (~100-200) but I read on TF documentation it's recommended to have 64
    # Then I noticed overfitting so I increased the sentence size to have more examples
    LARGEST_SENTENCE = 300
    X_train_ready, X_test_ready = pad_text(X_train_encoded, X_test_encoded, LARGEST_SENTENCE=LARGEST_SENTENCE)
    

	# 2. Train your network
    model, keras_callbacks = build_model(vocab_size, LARGEST_SENTENCE)
    
    # model automatically stops training if early stopping criteria are met
    history = model.fit(X_train_ready, y_train,validation_data=(X_test_ready, y_test),
                    batch_size = 100, epochs=2)#, callbacks=keras_callbacks)


    print('Final Training Accuracy:')
    # please note I save the model with the lowest validation loss
    # This is why I find the training accuracy of the epoch with the lowest 
    # validation loss
    print(history.history['accuracy'][-1])

	# 3. Save your model
    # automatically  saves the best one because of the callback (if used)
    #now im not using the callback 
    model.save('models/20607230_NLP_model.h5')