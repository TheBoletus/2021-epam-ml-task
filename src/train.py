"""
Script for training a model of goods classifier.

Input dataset gets split into training and testing groups. The splitting
ration can be changed via command line. The randomization state for
the splitting can be changed as well, and each value will yield
a reproducible sequence.

Training pipeline consists of:
- TfidfVectorizer (term frequency–inverse document frequency)
- MultinomialNB (naive Bayes - probabilistic - classifier)

Trained model gets saved as 'classifier_X' where X is the randomization
index used during training.

"""

import argparse
import logging
import pickle

import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from src.util import DATASET_FOLDER, MODEL_FOLDER, set_logging


def get_config():
    """
    Utility function for setting up the ArgumentParser.
    Returns the parsed parameters.

    :return: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name',
                        help='Name of preprocessed dataset file at /dataset')
    parser.add_argument('-s', '--split_ratio', type=float,
                        help='Dataset split ration, i.e. 0.2', default=0.2)
    parser.add_argument('-r', '--randomization_index', type=int,
                        help='Dataset splitter random state index', default=0)
    return parser.parse_args()


def log_results(_predictions, test_labels):
    """
    Utility function for logging the results of trained model.
    The following data gets logged:
    - confusion matrix
    - classification report
    - accuracy score

    :param _predictions: sklearn.Pipeline
    :param test_labels: list[str]
    :return: None
    """
    numpy.set_printoptions(linewidth=150)
    logging.info('Confusion matrix:\n{}'.format(
        confusion_matrix(test_labels, _predictions)
    ))
    logging.info('Classification report\n{}'.format(
        classification_report(test_labels, _predictions)))
    logging.info(f'Accuracy: {accuracy_score(test_labels, _predictions)}')


if __name__ == '__main__':

    set_logging()
    cfg = get_config()

    dataset_name = f'{DATASET_FOLDER}/{cfg.dataset_name}'
    df_train = pd.read_csv(dataset_name, header=0, delimiter=';')
    train_data = df_train['description'].values
    train_labels = df_train['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        train_data,
        train_labels,
        test_size=cfg.split_ratio,
        random_state=cfg.randomization_index)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    model_name = f'{MODEL_FOLDER}/classifier_{cfg.randomization_index}'
    with open(model_name, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)

    predictions = model.predict(X_test)
    log_results(predictions, y_test)
