"""
Script for basic querying the model HTTP service.
"""
import argparse
import json

import requests


def get_config():
    """
    Utility function for setting up the ArgumentParser.
    Returns the parsed parameters.

    :return: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--description',
        type=str,
        help='Goods description for sending to the model HTTP service.')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = get_config()
    data = {'description': cfg.description}
    prediction = requests.post('http://127.0.0.1/predict', json=data)
    print(json.loads(prediction.text))
