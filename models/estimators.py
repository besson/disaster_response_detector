import pandas as pd
import numpy as np
from nltk import pos_tag

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.chunk import ne_chunk

import nltk

nltk.download(['maxent_ne_chunker', 'words'])


class DocLength(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_count = pd.Series(X).apply(lambda x: len(self.tokenizer(x)))
        return pd.DataFrame(X_count)


class NerExtractor(BaseEstimator, TransformerMixin):

    def extract_ner(self, text):
        # tokenize by sentences
        accepted_labels = ['GPE', 'ORGANIZATION', 'PERSON']
        sentence_list = sent_tokenize(text)
        labels = []

        for sentence in sentence_list:
            trees = ne_chunk(pos_tag(word_tokenize(sentence)))

            for tree in trees:
                try:
                    labels.append(tree.flatten().label())
                except:
                    pass

        df_dict = {}

        labels = np.unique(labels).tolist()

        for label in accepted_labels:
            df_dict[label] = int(label in labels)

        return df_dict

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        ner_serie = pd.Series(X).apply(self.extract_ner)
        return pd.DataFrame(ner_serie.values.tolist(), index=ner_serie.index)
