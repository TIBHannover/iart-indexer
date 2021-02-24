from datetime import datetime
import logging

from elasticsearch import Elasticsearch, exceptions

from iart_indexer.database.suggester import Suggester
from elasticsearch.helpers import bulk

# from elasticsearch_dsl import Search


class ElasticSearchSuggester(Suggester):
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        self.hosts = config.get("host", "localhost")
        self.port = config.get("port", 9200)

        self.es = Elasticsearch([{"host": self.hosts, "port": self.port}])

        self.index = config.get("suggester", "suggester")
        self.type = config.get("type", "_doc")

        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(
                index=self.index,
                body={
                    "mappings": {
                        "properties": {
                            "origin_completion": {"type": "completion"},
                            "meta_completion": {"type": "completion"},
                            "features_completion": {"type": "completion"},
                            "annotations_completion": {"type": "completion"},
                        }
                    }
                },
            )

    def bulk_insert(self, generator):
        def add_fields(generator):
            for x in generator:
                # logging.info(f"BULK: {x}")
                yield {"_id": x["id"], "_index": self.index, **x}

        bulk(client=self.es, actions=add_fields(generator))

    def update_entry(self, hash_id, meta=None, features=None, annotations=None, origins=None):
        body = {}

        if meta is not None:
            body.update({"meta_completion": meta})

        if features is not None:
            body.update({"features_completion": features})

        if annotations is not None:
            body.update({"annotations_completion": annotations})

        if origins is not None:
            body.update({"origin_completion": origins})

        self.es.index(index=self.index, doc_type=self.type, id=hash_id, body=body)

    def complete(self, query, size=5):
        results = self.es.search(
            index=self.index,
            body={
                "suggest": {
                    "origin_completion": {
                        "prefix": query,
                        "completion": {"field": "origin_completion", "skip_duplicates": True},
                    },
                    "meta_completion": {
                        "prefix": query,
                        "completion": {"field": "meta_completion", "skip_duplicates": True},
                    },
                    "features_completion": {
                        "prefix": query,
                        "completion": {"field": "features_completion", "skip_duplicates": True},
                    },
                    "annotations_completion": {
                        "prefix": query,
                        "completion": {"field": "annotations_completion", "skip_duplicates": True},
                    },
                }
            },
            size=size,
        )
        return_values = []
        for field, x in results["suggest"].items():
            if len(x) != 1:
                continue
            options_list = []
            for option in x[0]["options"]:
                options_list.append(option["text"])

            return_values.append({"type": field.replace("_completion", ""), "options": options_list})
        return return_values
