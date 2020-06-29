import numpy as np
import tensorflow as tf
from preprocess import remove_URL, remove_punct, remove_punct, remove_stop_words, remove_empty, remove_case_white_space
from preprocess import max_seq_length, tokenizer, decode
from tensorflow.keras.preprocessing.sequence import pad_sequences

CATEGORIES = ['positive', 'negative']


def process(predictions):
    predictions = np.array([remove_URL(text) for text in predictions])
    predictions = np.array([remove_punct(text) for text in predictions])
    predictions = np.array([remove_case_white_space(text)
                            for text in predictions])
    predictions = np.array([remove_stop_words(text) for text in predictions])
    predictions = remove_empty(predictions)

    # tokenize test data
    predictions = tokenizer.texts_to_sequences(predictions)
    # pad test data
    predictions = pad_sequences(
        predictions, maxlen=max_seq_length, padding='post', truncating='post'
    )
    return predictions


predictions = np.array(
    ['good thing up', 'sad covid made low', 'nice to see prices high', 'going to buy shares', 'wouldnt go spend on ford', 'corona effecting netflix'])

model = tf.keras.models.load_model(
    'Stock-RNN-E(64)x32xD0.5x2-time-00-02.model')

prediction = model.predict(process(predictions))
print(prediction)
