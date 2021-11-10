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
from collections import defaultdict

LABEL_MARKER = '__label__'
FILTER = True

# ToDo: add documentation
# ToDo: add command line parameters


def get_label(label):
    return label.replace(LABEL_MARKER, '').replace('_', ' ')


def is_label(label):
    return label.startswith(LABEL_MARKER)


def is_description(description):
    if FILTER:
        return description.isalpha()
    return True


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


def write_row(_file, row_data):
    label = row_data.get('labels')[0]
    description = row_data.get('desc')
    row = f'{" ".join(description)};{label}\n'
    _file.write(row)


def log_statistics(counted, written):
    # ToDo: change prints to logging
    print('Statistics on labels, encountered / written / percentage:')
    for item in counted.keys():
        pct = int(written.get(item) / counted.get(item) * 100)
        print(f'{item}: {counted.get(item)} / {written.get(item)} / {pct}')


class DatasetSplitter:
    distribution = (
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'test',
        'test'
    )

    def __init__(self):
        self.labels = defaultdict(int)

    def get_direction(self, label_name):
        dataset_block_name = self.distribution[self.labels[label_name]]
        self.labels[label_name] += 1
        if self.labels[label_name] > 9:
            self.labels[label_name] = 0
        return dataset_block_name


if __name__ == '__main__':
    labels_written = defaultdict(int)
    label_counter = defaultdict(int)
    splitter = DatasetSplitter()

    with open('../dataset/train.csv', 'w') as train, \
            open('../dataset/test.csv', 'w') as test:
        datasets = {
            'train': train,
            'test': test
        }

        records = [x for x in dataset_generator('../dataset/dataset.csv')
                   if is_enough_data(x)]
        for record in records:
            label_value = record.get('labels')[0]
            if (labels_written.get(label_value) is None)\
                    or labels_written.get(label_value) < 1200:
                write_row(datasets.get(splitter.get_direction(label_value)),
                          record)
                labels_written[label_value] += 1
            label_counter[label_value] += 1

    log_statistics(label_counter, labels_written)
