import math
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import exceptions

# from elasticsearch_dsl import Search

from indexer.database.database import Database


class ElasticSearchDatabase(Database):
    def __init__(self, config: dict = {}):
        self._es = Elasticsearch()

        self._index = config.get("index", "iart")
        self._type = config.get("type", "_doc")

        if not self._es.indices.exists(index=self._index):
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
                        "path": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                    }
                }
            }

            self._es.indices.create(index=self._index, body=request_body)

    def insert_entry(self, hash_id, doc):
        self._es.index(index=self._index, doc_type=self._type, id=hash_id, body=doc)

    def update_entry(self, hash_id, doc):
        self._es.update(index=self._index, doc_type=self._type, id=hash_id, body={"doc": doc})

    def get_entry(self, hash_id):
        return self._es.get(index=self._index, doc_type=self._type, id=hash_id)["_source"]

    def update_plugin(self, hash_id, plugin_name, plugin_version, plugin_type, annotations):
        entry = self.get_entry(hash_id=hash_id)
        if entry is None:
            # TODO logging
            return

        # convert protobuf to dict
        annotations_list = []
        for anno in annotations:
            annotation_dict = {}
            result_type = anno.WhichOneof("result")
            if result_type == "feature":
                binary = anno.feature.binary
                feature = list(anno.feature.feature)

                annotation_dict["value"] = feature

                hash_splits_list = []
                for x in range(4):
                    hash_splits_list.append(binary[x * len(binary) // 4 : (x + 1) * len(binary) // 4])
                annotation_dict["hash"] = {f"split_{i}": x for i, x in enumerate(hash_splits_list)}
                annotation_dict["type"] = anno.feature.type
            annotations_list.append(annotation_dict)

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
        self._es.index(index=self._index, doc_type=self._type, id=hash_id, body=entry)

    def search(self, meta=None, features=None, classifiers=None, sort=None, size=5):
        if not self._es.indices.exists(index=self._index):
            return []
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

        if not isinstance(features, (list, set)):
            features = [features]

        for feature in features:
            pass
            # must.append({'fuzzy': {'feature.annotations.hash.split_0': {'value': '0000011001100000', 'fuzziness': 0}}})
            # es.count(index='iart_2',
            #          body={
            #              'query': {
            #                  'bool': {
            #                      'should': [{
            #                          'fuzzy': {
            #                              'feature.annotations.hash.split_0': {
            #                                  'value': '0000011001100000',
            #                                  'fuzziness': 0
            #                              }
            #                          }
            #                      }, {
            #                          'fuzzy': {
            #                              'feature.annotations.hash.split_1': {
            #                                  'value': '0000011001100000',
            #                                  'fuzziness': 0
            #                              }
            #                          }
            #                      }],
            #                      'minimum_should_match': 4
            #                  }
            #              }
            #          })

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
            results = self._es.search(index=self._index, body=body, size=size)
            for x in results["hits"]["hits"]:
                yield x["_source"]
        except exceptions.NotFoundError:
            return []
        # self._es.update('')

    def all(self, pagesize=250, scroll_timeout="10m", **kwargs):
        if not self._es.indices.exists(index=self._index):
            return None
        is_first = True
        while True:
            # Scroll next
            if is_first:  # Initialize scroll
                result = self._es.search(index=self._index, scroll="1m", **kwargs, body={"size": pagesize})
                is_first = False
            else:
                result = self._es.scroll(body={"scroll_id": scroll_id, "scroll": scroll_timeout})
            scroll_id = result["_scroll_id"]
            hits = result["hits"]["hits"]
            # Stop after no more docs
            if not hits:
                break
            # Yield each entry
            yield from (hit["_source"] for hit in hits)

    def drop(self):
        self._es.indices.delete(index=self._index, ignore=[400, 404])
