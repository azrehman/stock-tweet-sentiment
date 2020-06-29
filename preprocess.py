import string
import re
import numpy as np
import pandas as pd

from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


print('begin processing...')


### data assembly ###

df = pd.read_csv('stock_data.csv',)
# shuffle rows in data frame
df = df.sample(frac=1).reset_index(drop=True)


# seperate into featureset and classes
x_data = df['Text'].to_numpy()
y_data = df['Sentiment'].astype(int).to_numpy()


def make_one_hot(data, n_classes, dtype='float32'):
    return np.eye(n_classes)[data].astype(dtype)


# convert -1 to 0 in classes
y_data[y_data == -1] = 0

# one hot encode classes
y_data = make_one_hot(y_data, 2)


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.25)

### data processing ###


# remove any URLs from text
def remove_URL(text):
    url = re.compile(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''')
    return url.sub(r'', text)


# remove punctuation from text except for + and - because relevant for stock sentiment
def remove_punct(text):
    punct = string.punctuation
    punct = punct.replace('+', '')
    punct = punct.replace('-', '')
    return text.translate(str.maketrans('', '', punct))


# remove case from text and extra white space
def remove_case_white_space(text):
    return text.lower().strip()


# remove stop words ex) for, is, my
def remove_stop_words(text):
    stops = get_stop_words('english') + ['user', '\n']
    stops.remove('down')
    stops.remove('should')
    # return np.array(list(filter(lambda word: word not in stops, arr)))
    return ' '.join([word for word in text.split()
                     if word not in stops])


# remove empty strings if left in for some reason
def remove_empty(arr):
    return arr[arr != '']


# use functions to preprocess data
X_train = np.array([remove_URL(text) for text in X_train])
X_test = np.array([remove_URL(text) for text in X_test])

X_train = np.array([remove_punct(text) for text in X_train])
X_test = np.array([remove_punct(text) for text in X_test])

X_train = np.array([remove_case_white_space(text) for text in X_train])
X_test = np.array([remove_case_white_space(text) for text in X_test])


X_train = np.array([remove_stop_words(text) for text in X_train])
X_test = np.array([remove_stop_words(text) for text in X_test])

# X_train = remove_stop_words(X_train)
# X_test = remove_stop_words(X_test)

X_train = remove_empty(X_train)
X_test = remove_empty(X_test)


max_seq_length = 30

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# tokenize training data
X_train = tokenizer.texts_to_sequences(X_train)
# pad training data
X_train = pad_sequences(
    X_train, maxlen=max_seq_length, padding='post', truncating='post'
)

# tokenize test data
X_test = tokenizer.texts_to_sequences(X_test)
# pad test data
X_test = pad_sequences(
    X_test, maxlen=max_seq_length, padding='post', truncating='post'
)

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])

# method to go from tokenized sequences back to words


def decode(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# # saving featuresets
# np.save('processed_data/X_train.npy', X_train)
# np.save('processed_data/X_test.npy', X_test)

# # saving classes
# np.save('processed_data/y_train.npy', y_train)
# np.save('processed_data/y_test.npy', y_test)
print('done processing')
