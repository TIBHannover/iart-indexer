FROM python:3.7

COPY /pyproject.toml /src /indexer/
RUN pip install /indexer
CMD python -m iart_indexer --mode server -v -c /config.json
