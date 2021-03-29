import logging
import uuid
import json
from typing import Dict, List


class Aggregator:
    def __init__(
        self,
        database,
    ):
        super().__init__()
        self.database = database

    def meta_text_count(self, query=None, field_name=None, size=5):
        if field_name is not None:
            body = {
                "aggs": {
                    "meta_nested": {
                        "nested": {"path": "meta"},
                        "aggs": {
                            "meta_name_filter": {
                                "filter": {"term": {"meta.name": field_name}},
                                "aggs": {
                                    "meta_name_filter_aggr": {
                                        "terms": {"size": size, "field": "meta.value_str.keyword"}
                                    }
                                },
                            }
                        },
                    }
                }
            }

            if query is not None:
                body["query"] = query

            aggr = self.database.raw_aggregate(body=body)
            # logging.info(aggr)
            for x in aggr["aggregations"]["meta_nested"]["meta_name_filter"]["meta_name_filter_aggr"]["buckets"]:
                yield {"name": x["key"], "value": x["doc_count"]}
        else:
            body = {
                "aggs": {
                    "meta_nested": {
                        "nested": {"path": "meta"},
                        "aggs": {"meta_name_filter_aggr": {"terms": {"size": size, "field": "meta.value_str.keyword"}}},
                    }
                }
            }

            if query is not None:
                body["query"] = query

            aggr = self.database.raw_aggregate(body=body)
            # logging.info(aggr)
            for x in aggr["aggregations"]["meta_nested"]["meta_name_filter_aggr"]["buckets"]:
                yield {"name": x["key"], "value": x["doc_count"]}

    def __call__(self, query, field_names, size=250):
        logging.info("Aggregator")
        # logging.info(query)
        # logging.info(field_names)

        aggregations = []
        for field_name in field_names:
            # logging.info(field_name)

            field_path = field_name.split(".")
            if len(field_path) == 1:
                if field_path[0] == "meta":
                    aggregation = list(self.meta_text_count(query, size=size))
                    if aggregation and len(aggregation) > 0:
                        aggregations.append({"field_name": field_name, "entries": aggregation})

                # elif field_path[0] == "classifier":
                #     term = {
                #         "multi_match": {
                #             "query": e["query"],
                #             "fields": "classifier_text",
                #         }
                #     }
                # elif field_path[0] == "origin":
                #     term = {
                #         "multi_match": {
                #             "query": e["query"],
                #             "fields": "origin_text",
                #         }
                #     }

            if len(field_path) == 2:

                if field_path[0] == "meta":
                    aggregation = list(self.meta_text_count(query, field_name=field_path[1], size=size))
                    if aggregation and len(aggregation) > 0:
                        aggregations.append({"field_name": field_name, "entries": aggregation})

                    # term = {
                    #     "nested": {
                    #         "path": "meta",
                    #         "query": {
                    #             "bool": {
                    #                 "must": [
                    #                     {"match": {f"meta.name": field_path[1]}},
                    #                     {"match": {f"meta.value_str": e["query"]}},
                    #                 ]
                    #             }
                    #         },
                    #     }
                    # }

                # if field_path[0] == "origin":
                #     term = {
                #         "nested": {
                #             "path": "origin",
                #             "query": {
                #                 "bool": {
                #                     "must": [
                #                         {"match": {f"origin.name": field_path[1]}},
                #                         {"match": {f"origin.value_str": e["query"]}},
                #                     ]
                #                 }
                #             },
                #         }
                #     }
        return aggregations

    # def meta_text_count(self, query, field_name, size=5):
    #     body = {
    #         "aggs": {
    #             "meta_nested": {
    #                 "nested": {"path": "meta"},
    #                 "aggs": {
    #                     "meta_name_filter": {
    #                         "filter": {"term": {"meta.name": field_name}},
    #                         "aggs": {
    #                             "meta_name_filter_aggr": {"terms": {"size": size, "field": "meta.value_str.keyword"}}
    #                         },
    #                     }
    #                 },
    #             }
    #         }
    #     }
    #     aggr = self.database.raw_aggregate(body=body)
    #     print(aggr)
    #     for x in aggr["aggregations"]["meta_nested"]["meta_name_filter"]["meta_name_filter_aggr"]["buckets"]:
    #         yield {"name": x["key"], "value": x["doc_count"]}

    # def origin_test_count(self, field_name, size=5):
    #     body = {
    #         "aggs": {
    #             "origin_nested": {
    #                 "nested": {"path": "origin"},
    #                 "aggs": {
    #                     "origin_name_filter": {
    #                         "filter": {"term": {"origin.name": field_name}},
    #                         "aggs": {
    #                             "origin_name_filter_aggr": {
    #                                 "terms": {"size": size, "field": "origin.value_str.keyword"}
    #                             }
    #                         },
    #                     }
    #                 },
    #             }
    #         }
    #     }
    #     aggr = self.database.raw_aggregate(body=body)
    #     for x in aggr["aggregations"]["origin_nested"]["origin_name_filter"]["origin_name_filter_aggr"]["buckets"]:
    #         yield {"name": x["key"], "value": x["doc_count"]}

    # def feature_count(self, size=5):
    #     # body = {
    #     #     "aggs": {
    #     #         "feature_nested": {
    #     #             "nested": {"path": "feature"},
    #     #             "aggs": {
    #     #                 "feature_annotations_nested": {
    #     #                     "nested": {"path": "feature.annotations"},
    #     #                     "aggs": {
    #     #                         "feature_type_aggr": {
    #     #                             "terms": {
    #     #                                 "size": size,
    #     #                                 "field": "feature.annotations.type.keyword",
    #     #                                 "script": {
    #     #                                     "source": "doc['feature']['plugin']+feature.annotations.type.keyword",
    #     #                                     "lang": "painless",
    #     #                                 },
    #     #                             }
    #     #                         }
    #     #                     },
    #     #                 }
    #     #             },
    #     #         }
    #     #     }
    #     # }
    #     body = {
    #         "aggs": {
    #             "feature_nested": {
    #                 "nested": {"path": "feature"},
    #                 "aggs": {
    #                     "feature_annotations_nested": {
    #                         "terms": {
    #                             "size": size,
    #                             "field": "feature.plugin.keyword",
    #                         }
    #                     }
    #                 },
    #             }
    #         }
    #     }
    #     aggr = self.database.raw_aggregate(body=body)
    #     for x in aggr["aggregations"]["feature_nested"]["feature_annotations_nested"]["buckets"]:
    #         yield {"name": x["key"], "value": x["doc_count"]}

    # def classifier_tag_count(self, size=5):
    #     body = {
    #         "aggs": {
    #             "classifier_nested": {
    #                 "nested": {"path": "classifier"},
    #                 "aggs": {
    #                     "classifer_annotations_nested": {
    #                         "nested": {"path": "classifier.annotations"},
    #                         "aggs": {
    #                             "classifier_type_aggr": {
    #                                 "terms": {"size": size, "field": "classifier.annotations.name.keyword"}
    #                             }
    #                         },
    #                     }
    #                 },
    #             }
    #         }
    #     }
    #     aggr = self.database.raw_aggregate(body=body)
    #     for x in aggr["aggregations"]["classifier_nested"]["classifer_annotations_nested"]["classifier_type_aggr"][
    #         "buckets"
    #     ]:
    #         yield {"name": x["key"], "value": x["doc_count"]}
