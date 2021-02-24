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

    def meta_text_count(self, field_name, size=5):
        body = {
            "aggs": {
                "meta_nested": {
                    "nested": {"path": "meta"},
                    "aggs": {
                        "meta_name_filter": {
                            "filter": {"term": {"meta.name": field_name}},
                            "aggs": {
                                "meta_name_filter_aggr": {"terms": {"size": size, "field": "meta.value_str.keyword"}}
                            },
                        }
                    },
                }
            }
        }
        aggr = self.database.raw_aggregate(body=body)
        print(aggr)
        for x in aggr["aggregations"]["meta_nested"]["meta_name_filter"]["meta_name_filter_aggr"]["buckets"]:
            yield {"name": x["key"], "value": x["doc_count"]}

    def origin_test_count(self, field_name, size=5):
        body = {
            "aggs": {
                "origin_nested": {
                    "nested": {"path": "origin"},
                    "aggs": {
                        "origin_name_filter": {
                            "filter": {"term": {"origin.name": field_name}},
                            "aggs": {
                                "origin_name_filter_aggr": {
                                    "terms": {"size": size, "field": "origin.value_str.keyword"}
                                }
                            },
                        }
                    },
                }
            }
        }
        aggr = self.database.raw_aggregate(body=body)
        for x in aggr["aggregations"]["origin_nested"]["origin_name_filter"]["origin_name_filter_aggr"]["buckets"]:
            yield {"name": x["key"], "value": x["doc_count"]}

    def feature_count(self, size=5):
        # body = {
        #     "aggs": {
        #         "feature_nested": {
        #             "nested": {"path": "feature"},
        #             "aggs": {
        #                 "feature_annotations_nested": {
        #                     "nested": {"path": "feature.annotations"},
        #                     "aggs": {
        #                         "feature_type_aggr": {
        #                             "terms": {
        #                                 "size": size,
        #                                 "field": "feature.annotations.type.keyword",
        #                                 "script": {
        #                                     "source": "doc['feature']['plugin']+feature.annotations.type.keyword",
        #                                     "lang": "painless",
        #                                 },
        #                             }
        #                         }
        #                     },
        #                 }
        #             },
        #         }
        #     }
        # }
        body = {
            "aggs": {
                "feature_nested": {
                    "nested": {"path": "feature"},
                    "aggs": {
                        "feature_annotations_nested": {
                            "terms": {
                                "size": size,
                                "field": "feature.plugin.keyword",
                            }
                        }
                    },
                }
            }
        }
        aggr = self.database.raw_aggregate(body=body)
        for x in aggr["aggregations"]["feature_nested"]["feature_annotations_nested"]["buckets"]:
            yield {"name": x["key"], "value": x["doc_count"]}

    def classifier_tag_count(self, size=5):
        body = {
            "aggs": {
                "classifier_nested": {
                    "nested": {"path": "classifier"},
                    "aggs": {
                        "classifer_annotations_nested": {
                            "nested": {"path": "classifier.annotations"},
                            "aggs": {
                                "classifier_type_aggr": {
                                    "terms": {"size": size, "field": "classifier.annotations.name.keyword"}
                                }
                            },
                        }
                    },
                }
            }
        }
        aggr = self.database.raw_aggregate(body=body)
        for x in aggr["aggregations"]["classifier_nested"]["classifer_annotations_nested"]["classifier_type_aggr"][
            "buckets"
        ]:
            yield {"name": x["key"], "value": x["doc_count"]}
