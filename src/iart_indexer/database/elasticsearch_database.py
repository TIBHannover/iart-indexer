import math
import json
from datetime import datetime

from elasticsearch import Elasticsearch, exceptions
from elasticsearch.helpers import bulk


from iart_indexer.database.database import Database

import random
import uuid

# from elasticsearch_dsl import Search


class ElasticSearchDatabase(Database):
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        self.hosts = config.get("host", "localhost")
        self.port = config.get("port", 9200)

        self.es = Elasticsearch([{"host": self.hosts, "port": self.port}])

        self.index = config.get("index", "iart")
        self.type = config.get("type", "_doc")

        if not self.es.indices.exists(index=self.index):
            # pass
            request_body = {
                "mappings": {
                    "properties": {
                        "classifier": {
                            "type": "nested",
                            "properties": {
                                "annotations": {
                                    "type": "nested",
                                    "properties": {
                                        "name": {
                                            "type": "text",
                                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                        },
                                        "type": {
                                            "type": "text",
                                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                        },
                                        "value": {"type": "float"},
                                    },
                                },
                                "plugin": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "version": {"type": "float"},
                            },
                        },
                        "feature": {
                            "type": "nested",
                            "properties": {
                                "annotations": {
                                    "type": "nested",
                                    "properties": {
                                        "hash": {
                                            "properties": {
                                                "split_0": {
                                                    "type": "text",
                                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                                },
                                                "split_1": {
                                                    "type": "text",
                                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                                },
                                                "split_2": {
                                                    "type": "text",
                                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                                },
                                                "split_3": {
                                                    "type": "text",
                                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                                },
                                            }
                                        },
                                        "type": {
                                            "type": "text",
                                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                        },
                                        "value": {"type": "float"},
                                    },
                                },
                                "plugin": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "version": {"type": "float"},
                            },
                        },
                        "filename": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                        "id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                        "image": {"properties": {"height": {"type": "long"}, "width": {"type": "long"}}},
                        "meta": {
                            "properties": {
                                "artist_hash": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "artist_name": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "institution": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "location": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "title": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "yaer_max": {"type": "long"},
                                "year_min": {"type": "long"},
                            }
                        },
                        "origin": {
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "license": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                                "link": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                                },
                            }
                        },
                        "path": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                    }
                }
            }

            self.es.indices.create(index=self.index, body=request_body)

    def bulk_insert(self, generator):
        def add_fields(generator):
            for x in generator:
                yield {"_id": x["id"], "_index": self.index, **x}

        bulk(client=self.es, actions=add_fields(generator))

    def insert_entry(self, hash_id, doc):
        self.es.index(index=self.index, doc_type=self.type, id=hash_id, body=doc)

    def update_entry(self, hash_id, doc):
        self.es.update(index=self.index, doc_type=self.type, id=hash_id, body={"doc": doc})

    def get_entry(self, hash_id):
        try:
            return self.es.get(index=self.index, doc_type=self.type, id=hash_id)["_source"]
        except exceptions.NotFoundError:
            return None

    def get_entries(self, hash_ids):
        try:
            body = {"query": {"ids": {"type": "_doc", "values": hash_ids}}}
            print("#########################")
            print(body)
            results = self.es.search(index=self.index, doc_type=self.type, body=body, size=len(hash_ids))
            print(len(hash_ids))
            print(len(results["hits"]["hits"]))
            for x in results["hits"]["hits"]:
                yield x["_source"]
        except exceptions.NotFoundError:
            return None

    def update_plugin(self, hash_id, plugin_name, plugin_version, plugin_type, annotations):
        entry = self.get_entry(hash_id=hash_id)
        if entry is None:
            # TODO logging
            return

        # convert protobuf to dict
        annotations_list = []
        for anno in annotations:
            result_type = anno.WhichOneof("result")
            if result_type == "feature":
                annotation_dict = {}
                binary = anno.feature.binary
                feature = list(anno.feature.feature)

                if len(feature) == 64:
                    annotation_dict["val_64"] = feature
                elif len(feature) == 128:
                    annotation_dict["val_128"] = feature
                elif len(feature) == 256:
                    annotation_dict["val_256"] = feature
                else:
                    annotation_dict["value"] = feature

                hash_splits_list = []
                for x in range(4):
                    hash_splits_list.append(binary[x * len(binary) // 4 : (x + 1) * len(binary) // 4])
                annotation_dict["hash"] = {f"split_{i}": x for i, x in enumerate(hash_splits_list)}
                annotation_dict["type"] = anno.feature.type

                annotations_list.append(annotation_dict)

            if result_type == "classifier":
                for concept in anno.classifier.concepts:
                    annotation_dict = {}
                    annotation_dict["name"] = concept.concept
                    annotation_dict["type"] = concept.type
                    annotation_dict["value"] = concept.prob

                    annotations_list.append(annotation_dict)

        # print(annotations_list)

        # exit()
        if plugin_type in entry:
            founded = False
            for i, plugin in enumerate(entry[plugin_type]):
                if plugin["plugin"] == plugin_name:
                    founded = True
                    if plugin["version"] < plugin_version:
                        entry[plugin_type][i] = {
                            "plugin": plugin_name,
                            "version": plugin_version,
                            "annotations": annotations_list,
                        }
            if not founded:
                entry[plugin_type].append(
                    {"plugin": plugin_name, "version": plugin_version, "annotations": annotations_list}
                )
        else:
            entry.update(
                {plugin_type: [{"plugin": plugin_name, "version": plugin_version, "annotations": annotations_list}]}
            )
        self.es.index(index=self.index, doc_type=self.type, id=hash_id, body=entry)

    def search(self, meta=None, features=None, classifiers=None, sort=None, size=5):
        if not self.es.indices.exists(index=self.index):
            return []
        print("#########################")
        print(f"{meta} {features} {classifiers} {sort} {size}")
        print("#########################")
        terms = []

        if meta is not None:
            terms.append(
                {
                    "multi_match": {
                        "query": meta,
                        "fields": ["meta.title", "meta.author", "meta.location", "meta.institution"],
                    }
                }
            )

        if classifiers is not None:

            if not isinstance(classifiers, (list, set)):
                classifiers = [classifiers]

            classifier_should = []
            for classifier in classifiers:
                if isinstance(classifier, dict):
                    pass
                else:
                    classifier_should.append({"match": {"classifier.annotations.name": classifier}})
            terms.append(
                {
                    "nested": {
                        "path": "classifier",
                        "query": {
                            "nested": {
                                "path": "classifier.annotations",
                                "query": {"bool": {"should": classifier_should}},
                            }
                        },
                    }
                }
            )

        if features is not None:

            if not isinstance(features, (list, set)):
                features = [features]

            for feature in features:
                hash_splits = []
                for a in feature["annotations"]:
                    for sub_hash_index, value in a["hash"].items():
                        hash_splits.append(
                            {
                                "fuzzy": {
                                    f"feature.annotations.hash.{sub_hash_index}": {
                                        "value": value,
                                        "fuzziness": int(feature["fuzziness"]) if "fuzziness" in feature else 2,
                                    },
                                }
                            }
                        )

                term = {
                    "nested": {
                        "path": "feature",
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"feature.plugin": feature["plugin"]}},
                                    {
                                        "nested": {
                                            "path": "feature.annotations",
                                            "query": {
                                                "bool": {
                                                    # "must": {"match": {"feature.plugin": "yuv_histogram_feature"}},
                                                    "should": hash_splits,
                                                    "minimum_should_match": int(feature["minimum_should_match"])
                                                    if "minimum_should_match" in feature
                                                    else 4,
                                                },
                                            },
                                        }
                                    },
                                ]
                            },
                        },
                    }
                }
                terms.append(term)

        body = {"query": {"bool": {"should": terms}}}
        if sort is not None:
            sort_list = []
            if not isinstance(sort, (list, set)):
                sort = [sort]
            for x in sort:
                if x == "classifier":
                    sort_list.append(
                        {
                            "classifier.annotations.value": {
                                "order": "desc",
                                "nested": {"path": "classifier", "nested": {"path": "classifier.annotations"}},
                            }
                        }
                    )

            body.update({"sort": sort_list})
        try:
            results = self.es.search(index=self.index, body=body, size=size)
            for x in results["hits"]["hits"]:
                yield x["_source"]
        except exceptions.NotFoundError:
            return []
        # self.es.update('')

    def raw_search(self, body, size=5):
        if not self.es.indices.exists(index=self.index):
            return []

        try:
            results = self.es.search(index=self.index, body=body, size=size)
            for x in results["hits"]["hits"]:
                yield x["_source"]
        except exceptions.NotFoundError:
            return []
        # self.es.update('')

    def all(self, pagesize=250, scroll_timeout="10m", **kwargs):
        if not self.es.indices.exists(index=self.index):
            return None
        is_first = True
        while True:
            # Scroll next
            if is_first:  # Initialize scroll
                result = self.es.search(index=self.index, scroll="1m", **kwargs, body={"size": pagesize})
                is_first = False
            else:
                result = self.es.scroll(body={"scroll_id": scroll_id, "scroll": scroll_timeout})
            scroll_id = result["_scroll_id"]
            hits = result["hits"]["hits"]
            # Stop after no more docs
            if not hits:
                break
            # Yield each entry
            yield from (hit["_source"] for hit in hits)

    def drop(self):
        self.es.indices.delete(index=self.index, ignore=[400, 404])
