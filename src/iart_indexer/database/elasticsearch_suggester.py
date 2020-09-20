from datetime import datetime

from elasticsearch import Elasticsearch, exceptions

from iart_indexer.database.suggester import Suggester

# from elasticsearch_dsl import Search


class ElasticSearchSuggester(Suggester):
    def __init__(self, config=None):

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
                            "meta_completion": {"type": "completion"},
                            "features_completion": {"type": "completion"},
                            "annotations_completion": {"type": "completion"},
                        }
                    }
                },
            )

    def update_entry(self, hash_id, meta=None, features=None, annotations=None):
        body = {}

        if meta is not None:
            body.update({"meta_completion": meta})

        if features is not None:
            body.update({"features_completion": features})

        if annotations is not None:
            body.update({"annotations_completion": annotations})

        self.es.index(index=self.index, doc_type=self.type, id=hash_id, body=body)

    def complete(self, query, size=5):
        results = self.es.search(
            index=self.index,
            body={
                "suggest": {
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
            print(len(x))
            if len(x) != 1:
                continue
            options_list = []
            for option in x[0]["options"]:
                options_list.append(option["text"])

            return_values.append({"type": field.replace("_completion", ""), "options": options_list})
        return return_values
