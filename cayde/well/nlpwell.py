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

    def createTfidfFeatures(self) -> List[str]:
        avail_columns = []

        for column in self._text_cols:
            pass

        return avail_columns

    # def createSvdfFeatures(self) -> List[str]:
    #     # avail_columns = []
        #
        # n_train = df[~df['target'].isnull()].shape[0]
        # n_test = df[df['target'].isnull()].shape[0]
        #
        # # check to see if the TF-IDF features have been generated
        # # if not, generate new ones
        #
        # tfidfGenerator = TfidfFeatureGenerator('tfidf')
        # featuresTrain = tfidfGenerator.read('train')
        # xHeadlineTfidfTrain, xBodyTfidfTrain = featuresTrain[0], featuresTrain[1]
        #
        # xHeadlineTfidf = xHeadlineTfidfTrain
        # xBodyTfidf = xBodyTfidfTrain
        # if n_test > 0:
        #     # test set is available
        #     featuresTest = tfidfGenerator.read('test')
        #     xHeadlineTfidfTest, xBodyTfidfTest = featuresTest[0], featuresTest[1]
        #     xHeadlineTfidf = vstack([xHeadlineTfidfTrain, xHeadlineTfidfTest])
        #     xBodyTfidf = vstack([xBodyTfidfTrain, xBodyTfidfTest])
        #
        # # compute the cosine similarity between truncated-svd features
        # svd = TruncatedSVD(n_components=50, n_iter=15)
        # xHBTfidf = vstack([xHeadlineTfidf, xBodyTfidf])
        # svd.fit(xHBTfidf)  # fit to the combined train-test set (or the full training set for cv process)
        # xHeadlineSvd = svd.transform(xHeadlineTfidf)
        #
        # xHeadlineSvdTrain = xHeadlineSvd[:n_train, :]
        # outfilename_hsvd_train = "train.headline.svd.pkl"
        # with open(outfilename_hsvd_train, "wb") as outfile:
        #     cPickle.dump(xHeadlineSvdTrain, outfile, -1)
        # print
        # 'headline svd features of training set saved in %s' % outfilename_hsvd_train
        #
        # if n_test > 0:
        #     # test set is available
        #     xHeadlineSvdTest = xHeadlineSvd[n_train:, :]
        #     outfilename_hsvd_test = "test.headline.svd.pkl"
        #     with open(outfilename_hsvd_test, "wb") as outfile:
        #         cPickle.dump(xHeadlineSvdTest, outfile, -1)
        #     print
        #     'headline svd features of test set saved in %s' % outfilename_hsvd_test
        #
        # xBodySvd = svd.transform(xBodyTfidf)
        # print
        # 'xBodySvd.shape:'
        # print
        # xBodySvd.shape
        #
        # xBodySvdTrain = xBodySvd[:n_train, :]
        # outfilename_bsvd_train = "train.body.svd.pkl"
        # with open(outfilename_bsvd_train, "wb") as outfile:
        #     cPickle.dump(xBodySvdTrain, outfile, -1)
        # print
        # 'body svd features of training set saved in %s' % outfilename_bsvd_train
        #
        # if n_test > 0:
        #     # test set is available
        #     xBodySvdTest = xBodySvd[n_train:, :]
        #     outfilename_bsvd_test = "test.body.svd.pkl"
        #     with open(outfilename_bsvd_test, "wb") as outfile:
        #         cPickle.dump(xBodySvdTest, outfile, -1)
        #     print
        #     'body svd features of test set saved in %s' % outfilename_bsvd_test
        #
        # simSvd = np.asarray(map(cosine_sim, xHeadlineSvd, xBodySvd))[:, np.newaxis]
        # print
        # 'simSvd.shape:'
        # print
        # simSvd.shape
        #
        # simSvdTrain = simSvd[:n_train]
        # outfilename_simsvd_train = "train.sim.svd.pkl"
        # with open(outfilename_simsvd_train, "wb") as outfile:
        #     cPickle.dump(simSvdTrain, outfile, -1)
        #
        # if n_test > 0:
        #     # test set is available
        #     simSvdTest = simSvd[n_train:]
        #     outfilename_simsvd_test = "test.sim.svd.pkl"
        #     with open(outfilename_simsvd_test, "wb") as outfile:
        #         cPickle.dump(simSvdTest, outfile, -1)
        #     print
        #     'svd sim. features of test set saved in %s' % outfilename_simsvd_test
        #
        # for column in self._text_cols:
        #     pass
        #
        # return avail_columns

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