FROM python:3.8.6

RUN pip install poetry
WORKDIR /code
COPY poetry.lock pyproject.toml /code/
COPY . /code

RUN poetry config virtualenvs.create false --local
RUN poetry install
RUN poetry build

