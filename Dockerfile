FROM python:3.9

RUN pip install poetry

COPY pyproject.toml ./
RUN poetry export -f requirements.txt  --output requirements.txt
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm

COPY /pyproject.toml /src /indexer/
RUN pip install /indexer
