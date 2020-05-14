from cayde.well import Well
from cayde.plugins.datagenerator import DataGenerator
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import bert

from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from keras.layers import concatenate
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 128

def _get_segments(sentences):
    sentences_segments = []
    for sent in sentences:
      temp = []
      i = 0
      for token in sent.split(" "):
        temp.append(i)
        if token == "[SEP]":
          i += 1
      sentences_segments.append(temp)
    return sentences_segments

def _get_inputs(row, _maxlen, tokenizer, use_keras_pad=False):

    maxqnans = np.int((_maxlen-20)/2)
    pattern = '[^\w\s]+|\n' # remove everything including newline (|\n) other than words (\w) or spaces (\s)
    
    sentences = [
        ("[CLS] " + " ".join(tokenizer.tokenize(row['head'])[:maxqnans]) + " [SEP] "
        + " ".join(tokenizer.tokenize(row['body'])[:maxqnans]) + " [SEP] "
        + " ".join(tokenizer.tokenize(row['stance'])[:10]) + " [SEP]"
        ) 
    ]
    

    #generate masks
    # bert requires a mask for the words which are padded. 
    # Say for example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]
    sentences_mask = [[1]*len(sent.split(" "))+[0]*(_maxlen - len(sent.split(" "))) for sent in sentences]
 
    #generate input ids  
    # if less than max length provided then the words are padded
    if use_keras_pad:
      sentences_padded = pad_sequences(sentences.split(" "), dtype=object, maxlen=10, value='[PAD]',padding='post')
    else:
      sentences_padded = [sent + " [PAD]"*(_maxlen-len(sent.split(" "))) if len(sent.split(" "))!=_maxlen else sent for sent in sentences ]

    sentences_converted = [tokenizer.convert_tokens_to_ids(s.split(" ")) for s in sentences_padded]
    
    #generate segments
    # for each separation [SEP], a new segment is converted
    sentences_segment = _get_segments(sentences_padded)

    genLength = set([len(sent.split(" ")) for sent in sentences_padded])

    if _maxlen < 20:
      raise Exception("max length cannot be less than 20")
    elif len(genLength)!=1: 
      print(genLength)
      raise Exception("sentences are not of same size")



    #convert list into tensor integer arrays and return it
    #return sentences_converted,sentences_segment, sentences_mask
    #return [np.asarray(sentences_converted, dtype=np.int32), 
    #        np.asarray(sentences_segment, dtype=np.int32), 
    #        np.asarray(sen tences_mask, dtype=np.int32)]
    return [np.array(tf.cast(sentences_converted,tf.int32))[0], np.array(tf.cast(sentences_segment,tf.int32))[0], np.array(tf.cast(sentences_mask,tf.int32))[0]]

def fnc_score(y_true, y_pred):
  y_true = tf.argmax(y_true, axis=1)
  y_pred = tf.argmax(y_pred, axis=1)

  # compute max_score = 0.25*unrelated + (agree+disagree+discuss)
  total_count = tf.cast(tf.size(y_true), dtype=tf.int64)
  unrelated_count = tf.math.reduce_sum(tf.cast(tf.equal(tf.constant(0, dtype=tf.int64),y_true), tf.int64))
  related_count = tf.math.subtract(total_count, unrelated_count)
  max_score = tf.math.add(tf.math.scalar_mul(0.25, tf.cast(unrelated_count, dtype=tf.float32)), tf.cast(related_count, dtype=tf.float32))

  # compute score
  unrelated_pred = tf.cast(tf.equal(tf.cast(0, dtype=tf.int64), y_pred), dtype=tf.int64)
  unrelated_true = tf.cast(tf.equal(tf.cast(0, dtype=tf.int64), y_true), dtype=tf.int64)
  correct_unrelated_count = tf.math.reduce_sum(tf.cast(tf.equal(unrelated_pred, unrelated_true), dtype=tf.int64))
  correct_unrelated_count_score = tf.math.scalar_mul(0.25, tf.cast(correct_unrelated_count, dtype=tf.float32))

  is_related_mask = tf.not_equal(tf.cast(0, dtype=tf.int64), y_pred)
  is_correct_mask = tf.equal(y_true, y_pred)

  combined_mask_correct_related = tf.logical_and(is_related_mask, is_correct_mask)
  correct_related_count = tf.math.reduce_sum(tf.cast(combined_mask_correct_related, dtype=tf.float32))
  correct_related_count_score = tf.math.scalar_mul(1.0, correct_related_count)

  is_related_true_mask = tf.not_equal(tf.cast(0, dtype=tf.int64), y_true)
  combined_mask_related = tf.logical_and(is_related_mask, is_related_true_mask)
  combined_mask_incorrect_related = tf.logical_and(combined_mask_related, tf.logical_not(combined_mask_correct_related))
  incorrect_related_count = tf.math.reduce_sum(tf.cast(combined_mask_incorrect_related, dtype=tf.float32))
  incorrect_related_count_score = tf.math.scalar_mul(0.25, incorrect_related_count)

  score = tf.math.add_n([correct_unrelated_count_score, correct_related_count_score, incorrect_related_count_score])
  return tf.math.divide(score, max_score)

def build_model_fullyconnected():
    """add pretrained bert model as a keras layer"""
    input_word_ids = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    _, sout = bert_layer([input_word_ids, input_masks, input_segments])
    X = GlobalAveragePooling1D()(sout)
    X = Dense(784, activation='relu')(X) 
    # X = Dense(784, activation='relu')(X) 
    output_= Dense(4, activation='sigmoid', name='output')(X)

    model = Model([input_word_ids, input_masks, input_segments],output_)
    print(model.summary())

    return model

bert_path = {
  "LARGE": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
  "SMALL": "https://tfhub.dev/google/small_bert/bert_uncased_L-4_H-256_A-4/1"
}['LARGE']

bert_layer = hub.KerasLayer(bert_path,trainable=True)
vocab_file1 = bert_layer.resolved_object.vocab_file.asset_path.numpy()
bert_tokenizer_tfhub = bert.bert_tokenization.FullTokenizer(vocab_file1, do_lower_case=True)

x = Well('data/training_data.csv')
x.fetch()
x.input_cols = ['head', 'body']
x.output_col = 'stance'

x._df['tokens'] = [x.LazyCell(_get_inputs, (row, 128, bert_tokenizer_tfhub)) for index, row in x._df.iterrows()]
x.lazy_cols = ['tokens']
x.input_cols = ['tokens']
x.output_cols = ['stance']
x.input_cols

model = build_model_fullyconnected(128)
model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics=[fnc_score])
categories = sorted([_ for _ in x.df['stance'].unique()])

for batch in x.chunkGenerator(chunk_size=1024, threads=False):
    batch.input_cols = batch.expandColumn('tokens')
    tokens_0 = batch.expandColumn('tokens_0')
    tokens_1 = batch.expandColumn('tokens_1')
    tokens_2 = batch.expandColumn('tokens_2')

    training_y = batch._df[batch.splitCategoricalData('stance', expectedCategories=categories, useInts=True)]
    
    training_x = [
        batch._df[tokens_0].to_numpy(),
        batch._df[tokens_1].to_numpy(),
        batch._df[tokens_2].to_numpy()
    ]
    model.fit(training_x, training_y)

import pickle
import io

with io.open("BERT_MODEL.PICKLE", "wb") as handle:
  pickle.dump(model, handle)