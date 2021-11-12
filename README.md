# 2021 EPAM Machine Learning task

## Contents

[Model details](#model-details)

[Training the model](#training-the-model)

[Running the Streamlit service](#running-the-streamlit-service)

[Running the HTTP service](#running-the-http-service)


## Model details

The current classification model was trained using:
* TFIDF as a term-weighting scheme for the dataset
* Naive Bayes classifier from `sklearn`.

Below goes the testing report for `model/classifier_24` committed
to the repository.

### Accuracy

0.9085144927536232

### Confusion matrix 

```
[[228   1   1   0   1   0   0   0   2   4   0   0   6   0   0   0   0   0   2   0   0   0   0]
 [  0 211   2   0   0   0   1   0   1   0   1   0   0   1   0   0   7   1   0   0   0   0   0]
 [  3   4 212   0   2   0   0   0   5   2   2   0   0   1   2   2   7   1   1   1   0   0   0]
 [  0   0   0 231   0   0   0   0   0   0   0   0   1   0   0   1   0   0   1   0   0   3   0]
 [  1   2   4   1 176   0   4   6   3   7   6   1   1   7   3   7   2   3   0   5   0   0   0]
 [  1   4   0   5   0 217   4   2   0   1   2   2   1   1   3   0   1   0   1   1   3   3   1]
 [  2   7   3   0   5   1 179  10   9   1   0   0   0   3   3   3   1   1   2   1   1   3   1]
 [  0   0   0   0   7   1   5 200   0   1   0   1   0   2   1   3   1   1   0   2   1   0   0]
 [  0   0   3   0   5   0   5   0 215   0   5   3   2   0   2   2   2   0   5   0   0   0   1]
 [  0   0   0   0   1   0   0   0   0 222   0   0   1   0   0   0   0   0   0   2   0   0   0]
 [  0   0   2   0   3   0   0   0   2   4 213   1   6   2   1   2   0   1   2   0   0   0   0]
 [  0   0   0   0   2   0   0   0   0   2   1 238   1   0   0   0   0   0   0   0   0   0   0]
 [  0   0   2   0   2   0   0   0   1   1   2   1 204   2   3   3   0   2   2   0   0   1   0]
 [  0   1   3   1  10   0   0   2   1   3   2   0   1 222   1   6   2   1   0   2   0   0   0]
 [  0   0   1   0   3   0   1   1   2   4   1   1   0   0 222   3   0   6   0   0   1   0   0]
 [  0   1   1   0   0   1   0   0   1   0   1   1   0   0   0 228   1   0   0   0   1   0   0]
 [  0   1   1   0   4   0   1   0   1   0   0   0   0   0   0   0 214   2   0   0   1   1   0]
 [  0   0   1   0   3   0   0   0   0   2   0   1   4   2   0   0   1 221   1   0   0   1   1]
 [  0   0   1   0   1   0   0   0   1   1   0   0   5   0   1   0   0   0 243   0   0   2   0]
 [  0   1   3   0   0   0   2   0   1   0   0   0   0   0   0   1   0   0   0 238   0   1   0]
 [  4   2   0   0   1   5   1   0   3   0   0   1   4   1   0   0   0   1   0   0 202  13   0]
 [  0   1   0   0   0   0   0   0   0   1   0   0   0   0   0   0   1   0   0   2   1 251   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 228]]

```

### Classification report

```
              precision    recall  f1-score   support

           0       0.95      0.93      0.94       245
           1       0.89      0.94      0.92       225
           2       0.88      0.87      0.87       245
           3       0.97      0.97      0.97       237
           4       0.78      0.74      0.76       239
           5       0.96      0.86      0.91       253
           6       0.88      0.76      0.82       236
           7       0.90      0.88      0.89       226
           8       0.87      0.86      0.86       250
           9       0.87      0.98      0.92       226
          10       0.90      0.89      0.90       239
          11       0.95      0.98      0.96       244
          12       0.86      0.90      0.88       226
          13       0.91      0.86      0.88       258
          14       0.92      0.90      0.91       246
          15       0.87      0.97      0.92       236
          16       0.89      0.95      0.92       226
          17       0.92      0.93      0.92       238
          18       0.93      0.95      0.94       255
          19       0.94      0.96      0.95       247
          20       0.96      0.85      0.90       238
          21       0.90      0.98      0.94       257
          22       0.98      1.00      0.99       228

    accuracy                           0.91      5520
   macro avg       0.91      0.91      0.91      5520
weighted avg       0.91      0.91      0.91      5520
```

## Training the model

1) Download the dataset from
[Kaggle](https://www.kaggle.com/andrewboroda/ozon-product-category).

2) Copy the dataset to `<project>/dataset/` or make a symlink there.
Let's assume the file is called `dataset.csv`.

3) Perform the pre-processing:

```bash
python -m src.dataset -d dataset.csv
```

The pre-processor will go through the original data and pick
20,000+ items storing them in another table `<project>/dataset/train.csv`.

4) Perform the training:

```bash
python -m src.train -d train.csv -r 24
```

It will train the model and get all encountered labels and store them at:
* `<project>/model/classifier_24` - the model
* `<project>/model/labels.json` - the labels

The `-r 24` is a randomization state index for dataset splitter of `sklearn`.
Whatever randomization index is used for training, the saved model file will
have it in its name. Current model is called `classifier_24` thus the index
was 24.

## Running the Streamlit service

To run the service go to the project's root folder and execute:

```bash
streamlit run src/app_stream.py
```

## Running the HTTP service

There are two ways to run the HTTP service: with or without Docker

### Runnin without Docker

To run the service go the project's root folder and execute:

```bash
python -m src.app_service
```

### Running with Docker

Go to the project's root folder and build the container:

```bash
docker build .
```

Get the newly built image's ID and start a container with it:

```bash
docker run -p 127.0.0.1:80:8080/tcp <IMAGE_ID>
```

There is a helper script for probing the running service.
Run it from the project's root folder as follows:

```bash
python -m src.app_service_probe -d "some goods description from the dataset"
```

It will send the POST request with given description to `/predict` and print
the response.
