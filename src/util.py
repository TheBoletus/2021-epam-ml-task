import json
import logging
import pickle

DATASET_FOLDER = './dataset'
MODEL_FOLDER = './model'


def set_logging():
    """
    Function configures the logging module, setting the level
    and format of messages.

    :return: None
    """
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO)


class Model:

    """
    Class for handling the classifier model.
    """

    def __init__(self, model_path, labels_path):
        self.model = self.load_model(model_path)
        self.labels = self.load_labels(labels_path)

    @staticmethod
    def load_labels(labels_path):
        """
        Loads the JSON with array of labels that the model was trained with.

        :param labels_path: str
        :return: JSON
        """
        with open(labels_path, 'r') as _l:
            return json.load(_l)

    @staticmethod
    def load_model(model_path):
        """
        Loads given model of classifier.

        :param model_path: str
        :return: binary
        """
        with open(model_path, 'rb') as _m:
            return pickle.load(_m)

    def get_prediction(self, description):
        """
        Method transfers given description to model, gets generated prediction
        and returns a text of label that the prediction corresponds to.

        :param description: str
        :return: str
        """
        prep_description = self.preprocess_description(description)
        result = self.model.predict([prep_description, ])
        return self.labels[int(result)]

    @staticmethod
    def preprocess_description(description):
        """
        Method pre-processes given description.
        Right now the only change is filtering out all non-alphabetic
        sequences.

        :param description: str
        :return: str
        """
        filtered = [word for word in description.split() if word.isalpha()]
        return ' '.join(filtered)
