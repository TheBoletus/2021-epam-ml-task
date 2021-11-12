FROM python:3.9-slim-buster AS classifier-task

WORKDIR /app
EXPOSE 8080/tcp

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD model /app/model
ADD src /app/src

CMD ["python3", "-m", "src.app_service"]