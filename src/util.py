import logging

DATASET_FOLDER = './dataset'
MODEL_FOLDER = './model'


def set_logging():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO)
