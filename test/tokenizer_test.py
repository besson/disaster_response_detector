import unittest
from models.train_classifier import tokenize


class TokenizerTest(unittest.TestCase):

    def test_simple_tokenization(self):
        expected = ['weather', 'update']
        self.assertEqual(expected, tokenize('Weather update'))

    def test_remove_stopwords(self):
        expected = ['hurricane', '42']
        self.assertEqual(expected, tokenize('Is the Hurricane 42 over or is it not over'))

    def test_remove_pontuaction(self):
        expected = ['hurricane']
        self.assertEqual(expected, tokenize('Is the Hurricane over or is it not over ?!  !'))

    def test_apply_lemmatization(self):
        expected = ['call', 'good', 'number']
        self.assertEqual(expected, tokenize('calling better numbers'))

    def test_tokenize_more_than_one_sentence(self):
        expected = ['hurricane', 'please', 'call', 'number']
        self.assertEqual(expected, tokenize('Is the Hurricane over or is it not over ?. Please call this number!'))


if __name__ == '__main__':
    unittest.main()
