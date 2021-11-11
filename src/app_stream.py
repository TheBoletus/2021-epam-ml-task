import json
import pickle

import streamlit as st


@st.cache
def load_labels():
    """
    Function loads the pre-generated list of all labels encountered
    in training dataset.
    :return: list[str]
    """
    with open('./model/labels.json', 'r') as _l:
        return json.load(_l)


@st.cache
def load_model():
    """
    Function loads the pre-trained model.
    :return: binary
    """
    with open('./model/classifier', 'rb') as _m:
        return pickle.load(_m)


def preprocess_input(input_text):
    """
    Function performs preprocessing on provided piece of text.
    It filters out all non-alphabetical sequences of characters
    and returns the resulting string.
    :param input_text: str
    :return: str
    """
    filtered = [word for word in input_text.split() if word.isalpha()]
    return ' '.join(filtered)


def get_prediction(input_text):
    """
    Function handles the predicting. It prepares the labels, the model,
    preprocesses the text input and returns the prediction result as
    a string representing certain label.
    :param input_text:
    :return: str
    """
    labels = load_labels()
    prediction_model = load_model()

    filtered_input = preprocess_input(input_text)
    prediction = prediction_model.predict([filtered_input, ])
    return labels[int(prediction)]


# Main app code goes below

header = st.container()
description = st.container()
model = st.container()
footer = st.container()


with header:
    st.title('EPAM ML task')
    st.markdown("""
    This is a simple application build with Streamlit for interacting
    with the model.

    The model was trained on part of the Ozon Product Category
    dataset from Kaggle
    [challenge](https://www.kaggle.com/andrewboroda/ozon-product-category).
    """)

with model:
    st.header('Prediction')
    st.markdown("""
    Type in a description similar to ones in the dataset
    and press Enter.""")
    input_col, result_col = st.columns(2)

    desc = input_col.text_input('Description:')
    result_col.markdown('Predicted tag:')

    result_col.write(get_prediction(desc))

with footer:
    st.markdown('* * *')
    st.markdown('2021, Artem Tataurov')
    st.markdown(
        '[Code on GitHub](https://github.com/TheBoletus/2021-epam-ml-task)')
