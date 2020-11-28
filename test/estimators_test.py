import unittest
import pandas as pd
from pandas._testing import assert_frame_equal

from models.train_classifier import NerExtractor, DocLength
from models.train_classifier import tokenize


class EstimatorTest(unittest.TestCase):

    def test_doc_length(self):
        e = DocLength(tokenize)
        result = e.fit_transform('New York needs Peter Parker')
        expected = pd.DataFrame([5], index=[0])

        assert_frame_equal(expected, result)

    def test_ner_extractor(self):
        e = NerExtractor()
        result = e.fit_transform('New York needs Peter Parker')
        expected = pd.DataFrame([{'GPE': 1, 'ORGANIZATION': 0, 'PERSON': 1}], index=[0])

        assert_frame_equal(expected, result)

    def test_ner_extractor_with_multiple_sentences(self):
        e = NerExtractor()
        result = e.fit_transform('New York needs help. We need Peter Parker!')
        expected = pd.DataFrame([{'GPE': 1, 'ORGANIZATION': 0, 'PERSON': 1}], index=[0])

        assert_frame_equal(expected, result)

    def test_ner_extractor_for_no_entities(self):
        e = NerExtractor()
        result = e.fit_transform('Your are my friend')
        expected = pd.DataFrame([{'GPE': 0, 'ORGANIZATION': 0, 'PERSON': 0}], index=[0])

        assert_frame_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
