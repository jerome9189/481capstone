from cayde.well import Well

from cayde.well.exceptions import WellCacheException, DryWellException

from typing import List

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import gensim
from gensim import models
from gensim import downloader

import scipy

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import numpy as np
import re

BERT_MAX_SEQ_LENGTH = 128

__all__ = ["NLPWell"]

class NLPWell(Well):
    """A well with behavior altered to consider text data.
    """
    name = "NLPWell"

    _text_cols: List[str]


    def __init__(self, source, *args, **kwargs):
        super().__init__(source, *args, **kwargs)
        self._text_cols = []


    @property
    def text_cols(self) -> List[str]:
        "Labels what input column is what"
        if self._df is None:
            raise DryWellException()
        return self._text_cols[:]


    @text_cols.setter
    def text_cols(self, newColumns: List[str]):
        if self._df is None:
            raise DryWellException()
        for column in newColumns:
            if column not in self._input_cols:
                raise ValueError(f"{column} not an input column")
        self._text_cols = newColumns[:]

    def getToySample(self, *args, **kwargs) -> 'Well':
        well: NLPWell = super().getToySample(*args, **kwargs)
        well._text_cols = self._text_cols
        return well

    def clean_text(self, removeStopWords: bool = True, stemWords: bool = True) -> List[str]:
        "Uses nltk's word_tokenize to clean up text data"
        avail_columns = []
        for text_col in self._text_cols:
            self._df[f'{text_col}_1gram'] = self._df.apply(
                lambda row: preprocess_data(row[text_col], removeStopWords, stemWords), 
                axis=1
            )

            self._df[f'{text_col}_clean'] = self._df.apply(
                lambda row: " ".join(row[f"{text_col}_1gram"]), 
                axis=1
            )

            avail_columns.append(f'{text_col}_1gram')
            avail_columns.append(f'{text_col}_clean')

        return avail_columns


    def createNgrams(self, highestNgram: int = 4, countUnique: bool = True, recordSize: bool = True, recordRatio: bool = True) -> List[str]:
        if highestNgram < 1:
            raise ValueError("highestNgram must be in (1, 4]")

        avail_columns = []

        funcs = {
            2: getBigram,
            3: getTrigram,
            4: getQuadgram
        }

        for text_col in self._text_cols:
            for n in range(2, highestNgram + 1):
                self._df[f'{text_col}_{n}gram'] = self._df.apply(
                    lambda row: funcs[n](row[f'{text_col}_{n-1}gram'], '_'),
                    axis=1
                )
                avail_columns.append(f'{text_col}_{n}gram')

                if countUnique:
                    self._df[f'{text_col}_{n}gram_unique'] = self._df.apply(
                        lambda row: len(set(row[f'{text_col}_{n}gram'])),
                        axis=1
                    )
                    avail_columns.append(f'{text_col}_{n}gram_unique')
                
                if recordSize:
                    self._df[f'{text_col}_{n}gram_size'] = self._df.apply(
                        lambda row: len(row[f'{text_col}_{n}gram']),
                        axis=1
                    )

                    avail_columns.append(f'{text_col}_{n}gram_size')

                if recordRatio:
                    if countUnique:
                        uniques = self._df[f'{text_col}_{n}gram_unique']
                    else:
                        uniques = self._df.apply(
                            lambda row: len(set(row[f'{text_col}_{n}gram'])),
                            axis=1
                        )
                    
                    if recordSize:
                        sizes = self._df[f'{text_col}_{n}gram_size']
                    else:
                        sizes = self._df.apply(
                            lambda row: len(row[f'{text_col}_{n}gram']),
                            axis=1
                        )

                    self._df[f'{text_col}_{n}gram_unique_ratio'] = uniques / (sizes + 1)
                    avail_columns.append(f'{text_col}_{n}gram_unique_ratio')

        return avail_columns


    def createCountSentenceFeatures(self) -> List[str]:
        avail_columns = []

        for column in self._text_cols:
            self._df[f'{column}_sentence_count'] = self._df.apply(
                lambda row: len(sent_tokenize(row[f'{column}'])),
                axis=1
            )
            avail_columns.append(f'{column}_sentence_count')

        return avail_columns


    def createCountWordFeatures(self, words: List[str]) -> List[str]:
        avail_columns = []
        
        for column in self._text_cols:
            for word in words:
                self._df[f'{column}_contains_{word}'] = self._df.apply(
                    lambda row: word.lower() in row[column].lower(),
                    axis=1
                )
                avail_columns.append(f'{column}_contains_{word}')

        return avail_columns

    def createSvdfFeatures(self) -> List[str]:
        """Generates TF-IDF features for each text column in the well.
            And a pairwise cosine similarity between the TF-IDF and SVD features of each
            text column.

            Outputs a list of columns modified."""
        avail_columns = []

        # TODO: filter out stopwords or not? TF-IDF can work with both but one may be better. This one doesn't check for stopwords
        # 1). concatenate all the text columns to get a list containing all the raw text in the dataset
        all_text = []
        for column in self.text_cols:
            all_text.extend(list(self._df[f"{column}"]))

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(all_text)  # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        tf_idf_features = dict()
        for column in self.text_cols:
            # using the vocabulary found above, compute a TF-IDF vector for each text column
            column_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)

            # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
            tf_idf_features[f'{column}_tf_idf'] = column_vectorizer.fit_transform(self._df[f'{column}'])

        # 4). compute cosine similarity between headline tfidf features and body tfidf features
        for i in range(len(tf_idf_features.keys())):
            for j in range(i, len(tf_idf_features.keys())):
                if i != j:
                    vec1_name = list(tf_idf_features.keys())[i]
                    vec2_name = list(tf_idf_features.keys())[j]
                    simTfidf = np.asarray(map(sklearn.metrics.pairwise.cosine_similarity,
                                              tf_idf_features[vec1_name],
                                              tf_idf_features[vec2_name]))

                    self._df[f"{vec1_name}_cossim_{vec2_name}"] = simTfidf
                    avail_columns.append(f"{vec1_name}_cossim_{vec2_name}")


        # create truncated-svd features for each text column
        svd_features = dict()
        svd = sklearn.decomposition.TruncatedSVD(n_components=100, n_iter=15)

        for column in self.text_cols:
            # print(tf_idf_features[f"{column}_tf_idf"])
            # all_values = list(tf_idf_features.values())
            # all_values = list(map(np.array, all_values))
            # all_values = np.hstack(tuple(all_values))

            # The peculiar
            svd.fit(tf_idf_features[f'{column}_tf_idf'])
            # note that this turns a sparse matrix to an np array, which is expensive in terms of memory.
            # For larger bodies of text, this will be a memory-intensive operation
            svd_features[f'{column}_svd'] = svd.transform(tf_idf_features[f'{column}_tf_idf'].toarray())

        # compute the cosine similarity between truncated-svd features
        for i in range(len(svd_features.keys())):
            for j in range(i, len(svd_features.keys())):
                if i != j:
                    vec1_name = list(svd_features.keys())[i]
                    vec2_name = list(svd_features.keys())[j]
                    simSvd = np.asarray(map(sklearn.metrics.pairwise.cosine_similarity,
                                              svd_features[vec1_name],
                                              svd_features[vec2_name]))

                    self._df[f"{vec1_name}_cossim_{vec2_name}"] = simSvd
                    avail_columns.append(f"{vec1_name}_cossim_{vec2_name}")

        # add the SVD and TFIDF features to cayde
        for column in self._text_cols:
            self._df[f"{vec1_name}_cossim_{vec2_name}"] = simSvd
            for i in range(tf_idf_features[f'{column}_tf_idf'].shape[1]):
                avail_columns.append(f"{column}_tf_idf_{i}")
                self._df[f"{column}_tf_idf_{i}"] = tf_idf_features[f'{column}_tf_idf'][:, i]

            for i in range(svd_features[f'{column}_svd'].shape[1]):
                avail_columns.append(f"{column}_svd_{i}")
                self._df[f"{column}_svd_{i}"] = svd_features[f'{column}_svd'][:, i]

        return avail_columns

    def createWord2VecFeatures(self, modelLocation, keepUnigrams=False) -> List[str]:
        avail_columns = []

        model = gensim.models.KeyedVectors.load_word2vec_format(modelLocation, binary=True)

        for column in self._text_cols:

            if f"{column}_1gram" not in self._df.columns:
                # use regex to split each text column into words (unigrams)
                token_pattern = re.compile(r"(?u)\b\w\w+\b", flags=re.UNICODE)
                tokens = [x.lower() for x in token_pattern.findall(column)]

                if keepUnigrams:
                    self._df[f"{column}_1gram"] = self._df[f"{column}"].map(lambda x: token_pattern.findall(x))
                    unigrams = self._df[f"{column}_1gram"]
                    avail_columns.append(f"{column}_1gram")
                else:
                    unigrams = self._df[f"{column}"].map(lambda x: token_pattern.findall(x))
            else:
                unigrams = self._df[f"{column}_1gram"]

            # document vector built by adding together all the word vectors
            # using Google's pre-trained word vectors
            def word2Vec(word):
                if word in model:
                    return model[word]
                else:
                    return np.array([0.] * 300)

            text_vec = unigrams.map(lambda x: list(map(word2Vec, x)))

            # sum the words and normalize the vector
            text_vec = text_vec.map(lambda x: sum(x))
            text_vec = text_vec.map(lambda x: normalize(x.reshape(1, -1), axis=1))

            for i in range(300):
                # add a word2vec feature to the well
                new_column = []
                text_vec.apply(lambda x: new_column.append(x[0][i]))
                self._df[f"{column}_word2vec_f{i}"] = new_column
                avail_columns.append(f"{column}_word2vec_f{i}")


        return avail_columns
    def createBERTEncodings(self,
        autoAddColumns: bool = False, 
        tokenizerModelHub: str = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    ) -> List[str]:
        avail_columns = []

        tokenizer = create_tokenizer_from_hub_module(tokenizerModelHub)

        for text_col in self._text_cols:
            primedInputs = self._df.apply(
                lambda row: bert.run_classifier.InputExample(
                    guid=None, # Globally unique ID for bookkeeping, unused in this example
                    text_a = row[text_col], 
                    label = row[self._output_col]
                ), 
                axis = 1
            )

            self._df[f'{text_col}_bert'] = bert.run_classifier.convert_examples_to_features(
                primedInputs, 
                list(self._df[self._output_col].unique()),
                BERT_MAX_SEQ_LENGTH, 
                tokenizer
            )

            avail_columns.append(f'{text_col}_bert')

        return avail_columns



# Methods shown by Google Tensorflow Tutorials

def create_tokenizer_from_hub_module(bert_model_hub: str) -> bert.tokenization.FullTokenizer:
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
              vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

# Methods used from Talos In the News FNC Challenge

def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def getQuadgram(words, join_string):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny', 'boy']
        Output: a list of trigram, e.g., ['I_am_Denny_boy']
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in range(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as bigram
        lst = getTrigram(words, join_string)
    return lst

def getTrigram(words, join_string, skip=0):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of trigram, e.g., ['I_am_Denny']
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        lst = getBigram(words, join_string, skip)
    return lst

def getBigram(words, join_string, skip=0):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of bigram, e.g., ['I_am', 'am_Denny']
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        lst = words[:]

    return lst

def preprocess_data(line, removeStopWords = True, stemWords = True):
    # Credit: Talos in the News
    token_pattern = re.compile(r"(?u)\b\w\w+\b", flags = re.UNICODE)
    tokens = [x.lower() for x in token_pattern.findall(line)]

    if stemWords:
        english_stemmer = SnowballStemmer('english')
        tokens = [english_stemmer.stem(token) for token in tokens]

    if removeStopWords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tokens = [x for x in tokens if x not in stopwords]

    return tokens