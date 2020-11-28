import pandas as pd
import numpy as np
import nltk
import logging
import pickle
import sys

from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize, RegexpTokenizer, word_tokenize
from nltk.corpus.reader.wordnet import NOUN, ADJ, VERB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker', 'words'])
stop_words = set(stopwords.words('english'))
logging.basicConfig(filename='logs/cv_search.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)

NUM_CLASSES = 36


# CUSTOM ESTIMATORS
class DocLength(BaseEstimator, TransformerMixin):
    """ Estimator used to calculate document length (number of normalized tokens)

        Notes:
        In the initialization, it receives an tokenizer function that break document text
        and sentences into a list of tokens
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Calculate document length
        INPUT
            X - An iterable which yields either str, unicode or file objects, iterable
        OUTPUT
            X : array of shape (n_samples, 1), int
        """
        X_count = pd.Series(X).apply(lambda x: len(self.tokenizer(x)))
        return pd.DataFrame(X_count)


class NerExtractor(BaseEstimator, TransformerMixin):
    """ Estimator extract and attribute 'GPE', 'ORGANIZATION' or 'PERSON' named entities"""

    def _extract_ner(self, text: str) -> pd.Series:
        """
        Process data and extract named entities
        INPUT
            text - content in sentences to be processed, str
        OUTPUT
            extracted data : object with result of entity recognition process, pd.Series
        """
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
        """
        Assign named entity to each transformed sample
        INPUT
            X - An iterable which yields either str, unicode or file objects, iterable
        OUTPUT
            X : array of shape (n_samples, 3), int
        """
        ner_serie = pd.Series(X).apply(self._extract_ner)
        return pd.DataFrame(ner_serie.values.tolist(), index=ner_serie.index)


def load_data(database_filepath: str):
    """
    Load data from database
    INPUT
        database_filepath - path where sqlite database is saved, str
    OUTPUT
        X - dataset with independent variables (messagae), np.array
        Y - dataset with dependent variables (category names), np.array
        category_name - labels of Y, List
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('message_with_category', engine)

    X = df.message.values
    Y = df.iloc[:, -NUM_CLASSES:].values
    category_names = df.iloc[:, -NUM_CLASSES:].columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize words from input sentences
    INPUT
        text - message content, str
    OUTPUT
        cleaned tokens - cleaned tokens after tokenization phase, List
    """
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    word_tokenizer = RegexpTokenizer(r'\w+')

    clean_tokens = []

    for sentence in sentences:
        tokens = [word.lower() for word in word_tokenizer.tokenize(sentence)]
        filtered_tokens = filter(lambda x: x not in stop_words, tokens)

        for token in filtered_tokens:
            clean_token = lemmatizer.lemmatize(token, pos=NOUN)
            clean_token = lemmatizer.lemmatize(clean_token, pos=ADJ)
            clean_token = lemmatizer.lemmatize(clean_token, pos=VERB)
            clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
    Build a classification model:
      1. Create Machine Learning pipeline
      2. Run GridSearch
      3. Return best model
    INPUT
        None
    OUTPUT
        model - best trained model, ClassifierMixin
    """

    random_forest = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_extract', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())])),
            ('doc_length', DocLength(tokenizer=tokenize)),
            ('ner_extract', NerExtractor())])),
        ('clf', MultiOutputClassifier(estimator=random_forest, n_jobs=-1))
    ])

    parameters = {
        'features__text_extract__vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [100, 200]
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=-1, scoring='f1_macro')


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating the model using test data and display for each class:
      - precision
      - recall
      - f1-score
    INPUT
        model - trained classification model, ClassifierMixin
        X_test - test dataset with dependent variables (message), np.array
        Y_test - test dataset with indepdent variables (categories), np.array
    OUTPUT
        model - best trained model, ClassifierMixin
    """
    reports = []
    Y_pred = model.predict(X_test)

    for idx, target in enumerate(category_names):
        reports.append(classification_report(Y_test[:, idx].tolist(),
                                             Y_pred[:, idx].tolist(),
                                             labels=[0],
                                             target_names=[target]))

    for report in reports:
        logging.info('----------------------------------------------------')
        logging.info(report)


def save_model(model, model_filepath):
    """
    Save model into a pickle file
    INPUT
        model - trained classification model, ClassifierMixin
        model_filepath - file path to store the serialized model, str
    OUTPUT
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        logging.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        logging.info('Building model...')
        model = build_model()

        logging.info('Training model...')
        model.fit(X_train, Y_train)

        logging.info('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        logging.info('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        logging.info('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
