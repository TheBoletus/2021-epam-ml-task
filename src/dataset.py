"""
Script for pre-processing the dataset. The pre-processed data gets written
to `dataset/train.csv`, and the list of encountered labels gets written
as JSON to `model/labels.json`.

Initial dataset is located at
https://www.kaggle.com/andrewboroda/ozon-product-category
It contains 1,400,000 rows of a table. Each row contains one or more
columns with labels (text starting with `__label__`) and a description.

After excluding alphanumeric sequences from descriptions it will contain:
- 1,336,856 rows with a single label
- 61,016 rows with two labels
- 1,971 rows with three labels
- 9 rows with more than three labels

Only rows fitting the following requirements will be used for learning
and testing:
- has single label
- has no less than two word in description.
"""
import argparse
import json
import logging
from collections import defaultdict

from src.util import DATASET_FOLDER, MODEL_FOLDER, set_logging

LABEL_MARKER = '__label__'


def get_config():
    """
    Utility function for setting up the ArgumentParser.
    Returns the parsed parameters.

    :return: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset_name',
        help='Name of dataset file at /dataset that needs preprocessing')
    return parser.parse_args()


def get_label(label):
    """
    Pre-processes given label, removing unneeded characters.

    :param label: str
    :return: str
    """
    return label.replace(LABEL_MARKER, '').replace('_', ' ')


def is_label(label):
    """
    Checks if given text looks like a label.
    Current requirement: text starts with '__lable__'

    :param label: str
    :return: bool
    """
    return label.startswith(LABEL_MARKER)


def is_description(description):
    """
    Checks if given piece of text looks like a valid part of description.
    Current requirement: text consist of alphabetic characters only.

    :param description: str
    :return: str
    """
    return description.isalpha()


def is_enough_data(data_item):
    """
    Checks if given record from dataset has enough data.
    Current requirements:
    - there is a single label corresponding to description
    - there are two words in description at least.

    :param data_item: dict
    :return: bool
    """
    return len(data_item.get('labels')) == 1 \
           and len(data_item.get('desc')) > 2


def dataset_generator(file_name):
    """
    Generator wrapping given dataset file. The file gets processed
    as text file even though it is supposed to be a CSV table.
    If a row fits the requirements it gets yielded to external code.

    :param file_name: str
    :return: dict
    """
    with open(file_name, 'r') as f:
        for line in f.readlines():
            _line = line.split()
            labels = [get_label(label) for label in _line if is_label(label)]
            description = [desc for desc in _line if is_description(desc)]
            if labels and description:
                yield {
                    'desc': description,
                    'labels': labels
                }


def write_row(_file, row_data, label_index):
    """
    Writes a pre-processed row of data to a given CSV file:
    - description at column 0
    - label index at column 1.

    :param _file: File
    :param row_data: str
    :param label_index: int
    :return: None
    """
    row = f'{" ".join(row_data)};{label_index}\n'
    _file.write(row)


def log_statistics(counted, written):
    """
    Logs the statistics when the dataset pre-processing is finished.
    The metrics are the following:
    - total amount of descriptions
    - amount of descriptions written to the train.csv table
    - percentage of written to total.

    :param counted: int
    :param written: int
    :return: None
    """
    logging.info('Statistics on labels, total / written / percentage:')
    for item in counted.keys():
        pct = int(written.get(item) / counted.get(item) * 100)
        logging.info('{}: {} / {} / {}'.format(
            item,
            counted.get(item),
            written.get(item),
            pct))


if __name__ == '__main__':
    set_logging()
    cfg = get_config()

    labels_index = []
    labels_written = defaultdict(int)
    label_counter = defaultdict(int)

    with open(f'{DATASET_FOLDER}/train.csv', 'w') as train:
        train.write('description;label\n')
        dataset_file = f'{DATASET_FOLDER}/{cfg.dataset_name}'
        records = [x for x in dataset_generator(dataset_file)
                   if is_enough_data(x)]
        for record in records:
            label_value = record.get('labels')[0]

            if label_value not in labels_index:
                labels_index.append(label_value)

            if (labels_written.get(label_value) is None)\
                    or labels_written.get(label_value) < 1200:
                index = labels_index.index(label_value)
                write_row(train, record.get('desc'), index)
                labels_written[label_value] += 1

            label_counter[label_value] += 1

    with open(f'{MODEL_FOLDER}/labels.json', 'w') as _l:
        _l.write(json.dumps(labels_index, indent=4))

    log_statistics(label_counter, labels_written)
