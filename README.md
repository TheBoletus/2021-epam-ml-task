# 2021 EPAM Machine Learning task

## Contents

[Training the model](#training-the-model)

[Running the Streamlit service](#running-the-streamlit-service)

[Running the HTTP service](#running-the-http-service)


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
