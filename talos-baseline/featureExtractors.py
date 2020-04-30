'''
All code ported from Talos to work with Python3.X + Sklearn
'''

import pandas as pd
import numpy as np
from helpers import try_divide, cosine_sim
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

class FeatureGenerator(object):
    _name: str

    def name(self):
        return self._name

    def process(self, data: pd.DataFrame, header):
        '''
            input:
                data: pandas dataframe
            generate features and save them into a pickle file
        '''
        raise NotImplementedError("Abstract!")

    def read(self, header):
        '''
            read the feature matrix from a pickle file
        '''
        raise NotImplementedError("Abstract!")


class CountFeatureGenerator(FeatureGenerator):
    _name = "countFeatureGenerator"

    def process(self, df):

        grams = ["unigrams", "bigrams", "trigrams"]
        feat_names = ["head", "body"]

        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
                    list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    list(map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)]))

        # overlapping n-grams count
        for gram in grams:
            df["count_of_head_%s_in_body" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["head_" + gram] if w in set(x["body_" + gram])]), axis=1))
            df["ratio_of_head_%s_in_body" % gram] = \
                list(map(try_divide, df["count_of_head_%s_in_body" % gram], df["count_of_head_%s" % gram]))

        # number of sentences in headline and body
        for feat_name in feat_names:
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))

        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        for rf in _refuting_words:
            df[f'{rf}_exist'] = rf in df['head']
            df[f'{rf}_exist'] = df[f'{rf}_exist'].astype(int)

        # dump the basic counting features into a file
        feat_names = [
            n for n in df.columns if (
                "count" in n \
                or "ratio" in n \
                or "len_sent" in n\
                or "_exists" in n
            )
        ]

        with open("train.countvector.basic.pkl", "wb") as handle:
            pickle.dump(feat_names, handle, -1)
            pickle.dump(df[feat_names], handle, -1)
        

    def read(self, header='train'):

        filename_bcf = "%s.countvector.basic.pkl" % header
        with open(filename_bcf, "rb") as infile:
            feat_names = pickle.load(infile)
            training_data = pickle.load(infile)

        return [training_data]


class TfidfFeatureGenerator(FeatureGenerator):
    _name = "tfidfFeatureGenerator"

    def process(self, df):
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["all_text"])
        vocabulary = vec.vocabulary_

        vec_header = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        head_tfidf = vec_header.fit_transform(df['head_clean'])

        with open("train.head.tfidf.pkl", "wb") as outfile:
            pickle.dump(head_tfidf, outfile, -1)

        vec_body = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        body_tfidf = vec_body.fit_transform(df['body_clean'])

        with open("train.body.tfidf.pkl", "wb") as outfile:
            pickle.dump(head_tfidf, outfile, -1)

        sim_tfidf = np.asarray(list(map(cosine_sim, head_tfidf, body_tfidf)))[:, np.newaxis]

        with open("train.sim.tfidf.pkl", "wb") as outfile:
            pickle.dump(sim_tfidf, outfile, -1)

    def read(self, header='train'):
        filename_htfidf = "%s.head.tfidf.pkl" % header
        with open(filename_htfidf, "rb") as infile:
            head_tfidf = pickle.load(infile)

        filename_btfidf = "%s.body.tfidf.pkl" % header
        with open(filename_btfidf, "rb") as infile:
            body_tfidf = pickle.load(infile)

        filename_simtfidf = "%s.sim.tfidf.pkl" % header
        with open(filename_simtfidf, "rb") as infile:
            sim_tfidf = pickle.load(infile)

        return [head_tfidf, body_tfidf, sim_tfidf]


class SvdFeatureGenerator(FeatureGenerator):
    _name = "svdFeatureGenerator"

    pass




#   Copyright 2017 Cisco Systems, Inc.
#  
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  
#     http://www.apache.org/licenses/LICENSE-2.0
#  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.