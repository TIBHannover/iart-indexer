FROM python:3.7

COPY /pyproject.toml /src /indexer/
RUN pip install /indexer

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm

CMD python -m iart_indexer --mode server -v -c /config.json
