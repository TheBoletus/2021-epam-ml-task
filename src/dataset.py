"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset_name',
        help='Name of dataset file at /dataset that needs preprocessing')
    return parser.parse_args()


def get_label(label):
    return label.replace(LABEL_MARKER, '').replace('_', ' ')


def is_label(label):
    return label.startswith(LABEL_MARKER)


def is_description(description):
    return description.isalpha()


def is_enough_data(data_item):
    return len(data_item.get('labels')) == 1 \
           and len(data_item.get('desc')) > 2


def dataset_generator(file_name):
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
    row = f'{" ".join(row_data)};{label_index}\n'
    _file.write(row)


def log_statistics(counted, written):
    logging.info('Statistics on labels, encountered / written / percentage:')
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
