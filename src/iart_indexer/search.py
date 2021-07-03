import logging
import uuid
import json

import numpy as np

from iart_indexer import indexer_pb2
from typing import Dict, List
from iart_indexer.utils import get_features_from_db_entry

from iart_indexer.utils import image_from_proto, dict_from_proto

import faulthandler

from elasticsearch_dsl import Search, Q

import time


class Searcher:
    def __init__(
        self,
        database,
        feature_plugin_manager=None,
        image_text_plugin_manager=None,
        classifier_plugin_manager=None,
        indexer_plugin_manager=None,
        mapping_plugin_manager=None,
        aggregator=None,
    ):
        super().__init__()
        self.database = database
        self.feature_plugin_manager = feature_plugin_manager
        self.image_text_plugin_manager = image_text_plugin_manager
        self.classifier_plugin_manager = classifier_plugin_manager
        self.indexer_plugin_manager = indexer_plugin_manager
        self.mapping_plugin_manager = mapping_plugin_manager
        self.aggregator = aggregator

    def merge_feature(self, features):
        plugin_feature = {}
        for p in features:
            plugin = p["plugin"]
            type = p["type"]
            value = p["value"]
            weight = p["weight"]
            if plugin not in plugin_feature:
                plugin_feature[plugin] = {}
            if type not in plugin_feature[plugin]:
                plugin_feature[plugin][type] = []
            plugin_feature[plugin][type].append({"value": np.asarray(value), "weight": weight})

        result_features = []
        for p, plugin_data in plugin_feature.items():

            for t, values in plugin_data.items():

                mean_value = np.mean(np.stack([v["value"] * v["weight"] for v in values]), axis=0)
                mean_weight = np.mean(np.stack([np.abs(v["weight"]) for v in values]))
                result_features.append({"plugin": p, "type": t, "value": mean_value.tolist(), "weight": mean_weight})

        return result_features

    def parse_query(self, query):
        logging.info(query)
        result = {}
        logging.info(query)
        text_search = []
        range_search = []
        feature_search = []
        aggregate_fields = []

        # random seed
        if query.random_seed is not None and str(query.random_seed) != "":
            seed = str(query.random_seed)
        else:
            seed = uuid.uuid4().hex

        # parsing whitelists
        whitelist = []
        for id in query.ids:
            whitelist.append(id)

        if len(whitelist) == 0:
            whitelist = None

        # parsing collection

        collections = query.collections
        include_default_collection = query.include_default_collection

        # Parse sorting args
        sorting = None
        logging.info(f"Sorting: {query.sorting}")
        if query.sorting == indexer_pb2.SearchRequest.SORTING_CLASSIFIER:
            sorting = "classifier"
        if query.sorting == indexer_pb2.SearchRequest.SORTING_FEATURE:
            sorting = "feature"
        if query.sorting == indexer_pb2.SearchRequest.SORTING_RANDOM:
            sorting = "random"
        if query.sorting == indexer_pb2.SearchRequest.SORTING_RANDOM_FEATURE:
            logging.info("SORTING_RANDOM_FEATURE")

            # Add a random feature query to terms if
            entry = list(self.database.get_random_entries(seed=seed, size=1))[0]
            sorting = "feature"
            random_term = query.terms.add()
            random_term.feature.image.id = entry["id"]
            feature_exist = False
            for term in query.terms:
                if term == random_term:
                    continue

                term_type = term.WhichOneof("term")
                if term_type == "feature":
                    feature_exist = True
                    for p in term.feature.plugins:
                        plugins = random_term.feature.plugins.add()
                        plugins.name = p.name
                        plugins.weight = p.weight

            if not feature_exist:
                plugins = random_term.feature.plugins.add()
                plugins.name = "clip_embedding_feature"
                plugins.weight = 1.0

        # Parse mapping args
        mapping = None
        if query.mapping == indexer_pb2.SearchRequest.MAPPING_UMAP:
            mapping = "umap"
        mapping_options = dict_from_proto(query.mapping_options)

        clustering = None
        if query.clustering == indexer_pb2.SearchRequest.CLUSTERING_KMEANS:
            clustering = "kmeans"
        if query.clustering == indexer_pb2.SearchRequest.CLUSTERING_AGGLOMERATIVE:
            clustering = "agglomerative"
        clustering_options = dict_from_proto(query.clustering_options)

        # Parse additional fields
        extras = []
        for extra in query.extras:
            if extra == indexer_pb2.SearchRequest.EXTRA_FEATURES:
                extras.append("features")

        for term in query.terms:

            term_type = term.WhichOneof("term")
            if term_type == "text":
                text = term.text
                field = text.field.lower()
                flag = "should"
                if text.flag == indexer_pb2.TextSearchTerm.MUST:
                    flag = "must"
                if text.flag == indexer_pb2.TextSearchTerm.SHOULD:
                    flag = "should"
                if text.flag == indexer_pb2.TextSearchTerm.NOT:
                    flag = "not"

                text_search.append({"field": field, "query": text.query, "flag": flag})

            if term_type == "number":
                number = term.number
                field = number.field.lower()

                # parse number query field
                query_type = number.WhichOneof("query")
                if query_type == "string_query":
                    try:
                        query_term = int(number.string_query)
                    except:
                        try:
                            query_term = float(number.string_query)
                        except:
                            continue
                if query_type == "int_query":
                    query_term = number.int_query
                if query_type == "float_query":
                    query_term = number.float_query

                # parse relation
                relation = "eq"
                if number.relation == indexer_pb2.NumberSearchTerm.GREATER:
                    relation = "gt"
                if number.relation == indexer_pb2.NumberSearchTerm.GREATER_EQ:
                    relation = "gte"
                if number.relation == indexer_pb2.NumberSearchTerm.EQ:
                    relation = "eq"
                if number.relation == indexer_pb2.NumberSearchTerm.LESS_EQ:
                    relation = "lte"
                if number.relation == indexer_pb2.NumberSearchTerm.LESS:
                    relation = "lt"

                flag = "should"
                if number.flag == indexer_pb2.NumberSearchTerm.MUST:
                    flag = "must"
                if number.flag == indexer_pb2.NumberSearchTerm.SHOULD:
                    flag = "should"
                if number.flag == indexer_pb2.NumberSearchTerm.NOT:
                    flag = "not"

                range_search.append({"field": field, "query": query_term, "relation": relation, "flag": flag})

            if term_type == "image_text":

                image_text = term.image_text
                feature_results = list(
                    self.image_text_plugin_manager.run(
                        [image_text.query]
                    )  # , plugins=[x.name.lower() for x in feature.plugins])
                )[0]
                if image_text.flag == indexer_pb2.ImageTextSearchTerm.NEGATIVE:
                    weight_mult = -1.0
                else:
                    weight_mult = 1.0

                for p in image_text.plugins:
                    # feature_results
                    for f in feature_results["plugins"]:
                        if p.name.lower() == f._plugin.name.lower():

                            annotations = []
                            for anno in f._annotations[0]:
                                result_type = anno.WhichOneof("result")
                                if result_type == "feature":
                                    annotation_dict = {}
                                    binary = anno.feature.binary
                                    feature = list(anno.feature.feature)

                                    feature_search.append(
                                        {
                                            "plugin": f._plugin.name,
                                            "type": anno.feature.type,
                                            "value": feature,
                                            "weight": p.weight * weight_mult,
                                        }
                                    )

            if term_type == "feature":
                feature = term.feature

                if feature.flag == indexer_pb2.FeatureSearchTerm.NEGATIVE:
                    weight_mult = -1.0
                else:
                    weight_mult = 1.0

                query_feature = []

                entry = None
                if feature.image.id is not None and feature.image.id != "":
                    entry = self.database.get_entry(feature.image.id)

                if entry is not None:
                    entry = get_features_from_db_entry(entry)

                    for p in feature.plugins:
                        if p.weight is None or abs(p.weight) < 1e-4:
                            continue
                        # TODO add weight
                        for f in entry["feature"]:

                            if p.name.lower() == f["plugin"].lower():

                                feature_search.append(
                                    {
                                        **f,
                                        "weight": p.weight * weight_mult,
                                    }
                                )

                # TODO add example search here
                else:

                    image = image_from_proto(feature.image)

                    feature_results = list(
                        self.feature_plugin_manager.run([image])  # , plugins=[x.name.lower() for x in feature.plugins])
                    )[0]

                    for p in feature.plugins:

                        if p.weight is None or abs(p.weight) < 1e-4:
                            continue
                        # feature_results
                        for f in feature_results["plugins"]:
                            if p.name.lower() == f._plugin.name.lower():

                                annotations = []
                                for anno in f._annotations[0]:
                                    result_type = anno.WhichOneof("result")
                                    if result_type == "feature":
                                        annotation_dict = {}
                                        binary = anno.feature.binary
                                        feature = list(anno.feature.feature)

                                        feature_search.append(
                                            {
                                                "plugin": f._plugin.name,
                                                "type": anno.feature.type,
                                                "value": feature,
                                                "weight": p.weight * weight_mult,
                                            }
                                        )

        result.update({"text_search": text_search})
        result.update({"feature_search": feature_search})
        result.update({"range_search": range_search})
        result.update({"sorting": sorting})
        result.update({"mapping": mapping})
        result.update({"mapping_options": mapping_options})
        result.update({"clustering": clustering})
        result.update({"clustering_options": clustering_options})
        result.update({"extras": extras})
        result.update({"seed": seed})
        result.update({"whitelist": whitelist})

        result.update({"collections": collections})
        result.update({"include_default_collection": include_default_collection})

        if len(query.aggregate.fields) and query.aggregate.size > 0:
            aggregate_fields = list(query.aggregate.fields)
            result.update(
                {
                    "aggregate": {
                        "fields": aggregate_fields,
                        "size": query.aggregate.size,
                        "use_query": query.aggregate.use_query,
                    }
                }
            )

        return result

    def build_search_body_old(
        self,
        text_search: List = [],
        range_search: List = [],
        feature_search: List = [],
        whitelist: List = [],
        sorting=None,
        seed=None,
    ):
        # Seed all random functions
        if seed is not None and str(seed) != "":
            seed = str(seed)
        else:
            seed = uuid.uuid4().hex

        must_terms = []
        should_terms = []
        for e in text_search:
            term = None
            if "field" not in e or e["field"] in [None, ""]:
                term = {
                    "multi_match": {
                        "query": e["query"],
                        "fields": ["meta_text", "classifier_text"],
                    }
                }
            else:
                field_path = e["field"].split(".")
                if len(field_path) == 1:
                    if field_path[0] == "meta":
                        term = {
                            "multi_match": {
                                "query": e["query"],
                                "fields": "meta_text",
                            }
                        }
                    elif field_path[0] == "classifier":
                        term = {
                            "multi_match": {
                                "query": e["query"],
                                "fields": "classifier_text",
                            }
                        }
                    elif field_path[0] == "origin":
                        term = {
                            "multi_match": {
                                "query": e["query"],
                                "fields": "origin_text",
                            }
                        }

                if len(field_path) == 2:

                    if field_path[0] == "meta":
                        term = {
                            "nested": {
                                "path": "meta",
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"match": {f"meta.name": field_path[1]}},
                                            {"match": {f"meta.value_str": e["query"]}},
                                        ]
                                    }
                                },
                            }
                        }

                    if field_path[0] == "origin":
                        term = {
                            "nested": {
                                "path": "origin",
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"match": {f"origin.name": field_path[1]}},
                                            {"match": {f"origin.value_str": e["query"]}},
                                        ]
                                    }
                                },
                            }
                        }

            if term is None:
                continue

            if "flag" in e and e["flag"] is not None:
                if e["flag"] == "must":
                    must_terms.append(term)
                else:
                    should_terms.append(term)
            else:
                should_terms.append(term)

        for e in range_search:
            term = None
            if "field" not in e or e["field"] is None:
                # There is no generic int field in the database structure
                continue
            else:
                field_path = e["field"].split(".")
                if len(field_path) == 2:

                    if field_path[0] == "meta":
                        name_match = {"match": {f"meta.name": field_path[1]}}

                        if e["relation"] != "eq":
                            value_match = {"range": {f"meta.value_int": {e["relation"]: e["query"]}}}
                        else:

                            value_match = {"term": {f"meta.value_int": {"value": e["query"]}}}

                        term = {
                            "nested": {
                                "path": "meta",
                                "query": {"bool": {"must": [name_match, value_match]}},
                            }
                        }

                    if field_path[0] == "origin":

                        name_match = {"match": {f"origin.name": field_path[1]}}

                        if e["relation"] != "eq":
                            value_match = {"range": {f"origin.value_int": {e["relation"]: e["query"]}}}
                        else:

                            value_match = {"term": {f"origin.value_int": {"value": e["query"]}}}

                        term = {
                            "nested": {
                                "path": "origin",
                                "query": {"bool": {"must": [name_match, value_match]}},
                            }
                        }

            if term is None:
                continue

            if "flag" in e and e["flag"] is not None:
                if e["flag"] == "must":
                    must_terms.append(term)
                else:
                    should_terms.append(term)
            else:
                should_terms.append(term)
        # for e in classifier_search:
        #     search_terms.append(
        #         {
        #             "nested": {
        #                 "path": "classifier",
        #                 "query": {
        #                     "nested": {
        #                         "path": "classifier.annotations",
        #                         "query": {"bool": {"should": [{"match": {"classifier.annotations.name": e["query"]}}]}},
        #                     }
        #                 },
        #             }
        #         }
        #     )

        elastic_sorting = []
        if isinstance(sorting, str) and sorting.lower() == "classifier":
            elastic_sorting.append(
                {
                    "classifier.annotations.value": {
                        "order": "desc",
                        "mode": "sum",
                        "nested": {"path": "classifier", "nested": {"path": "classifier.annotations"}},
                    }
                }
            )

        if isinstance(sorting, str) and sorting.lower() == "random":
            should_terms.append({"function_score": {"functions": [{"random_score": {"seed": seed}}]}})

        if whitelist is not None and len(whitelist) > 0:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"ids": {"type": "_doc", "values": whitelist}},
                            {"bool": {"should": should_terms, "must": must_terms}},
                        ]
                    }
                }
            }

        else:
            body = {"query": {"bool": {"should": should_terms, "must": must_terms}}}

        if elastic_sorting is not None:
            body.update({"sort": elastic_sorting})

        return body

    def build_search_body(
        self,
        text_search: List = [],
        range_search: List = [],
        feature_search: List = [],
        whitelist: List = [],
        sorting=None,
        seed=None,
    ):

        search = Search()
        # Seed all random functions
        if seed is not None and str(seed) != "":
            seed = str(seed)
        else:
            seed = uuid.uuid4().hex

        must_terms = []
        should_terms = []
        must_not_terms = []
        for e in text_search:
            term = None
            if "field" not in e or e["field"] in [None, ""]:
                term = Q("multi_match", fields=["meta_text", "classifier_text"], query=e["query"])
            else:
                field_path = e["field"].split(".")
                if len(field_path) == 1:
                    if field_path[0] == "meta":
                        term = Q("multi_match", fields=["meta_text"], query=e["query"])
                    elif field_path[0] == "classifier":
                        term = Q("multi_match", fields=["classifier_text"], query=e["query"])
                    elif field_path[0] == "origin":
                        term = Q("multi_match", fields=["origin_text"], query=e["query"])

                if len(field_path) == 2:

                    if field_path[0] == "meta":
                        term = Q(
                            "nested",
                            path="meta",
                            query=Q(
                                "bool",
                                must=[Q("match", meta__name=field_path[1]), Q("match", meta__value_str=e["query"])],
                            ),
                        )
                    if field_path[0] == "origin":
                        term = Q(
                            "nested",
                            path="origin",
                            query=Q(
                                "bool",
                                must=[Q("match", origin__name=field_path[1]), Q("match", origin__value_str=e["query"])],
                            ),
                        )

            if term is None:
                continue

            if "flag" in e and e["flag"] is not None:
                if e["flag"] == "must":
                    must_terms.append(term)
                elif e["flag"] == "should":
                    should_terms.append(term)
                else:
                    must_not_terms.append(term)
            else:
                should_terms.append(term)

        for e in range_search:
            term = None
            if "field" not in e or e["field"] is None:
                # There is no generic int field in the database structure
                continue
            else:
                field_path = e["field"].split(".")
                if len(field_path) == 2:

                    if field_path[0] == "meta":
                        name_match = Q("match", meta__name=field_path[1])

                        if e["relation"] != "eq":
                            value_match = Q("range", meta__value_int={e["relation"]: e["query"]})
                        else:
                            value_match = Q("term", meta__value_int=e["query"])

                        # value_match = {"term": {f"meta.value_int": {"value": e["query"]}}}

                        term = Q(
                            "nested",
                            path="meta",
                            query=Q(
                                "bool",
                                must=[name_match, value_match],
                            ),
                        )

                    if field_path[0] == "origin":
                        name_match = Q("match", origin__name=field_path[1])

                        if e["relation"] != "eq":
                            value_match = Q("range", origin__value_int={e["relation"]: e["query"]})
                        else:
                            value_match = Q("term", origin__value_int=e["query"])

                        # value_match = {"term": {f"meta.value_int": {"value": e["query"]}}}

                        term = Q(
                            "nested",
                            path="origin",
                            query=Q(
                                "bool",
                                must=[name_match, value_match],
                            ),
                        )

            if term is None:
                continue

            if "flag" in e and e["flag"] is not None:
                if e["flag"] == "must":
                    must_terms.append(term)
                elif e["flag"] == "should":
                    should_terms.append(term)
                else:
                    must_not_terms.append(term)
            else:
                should_terms.append(term)

        if isinstance(sorting, str) and sorting.lower() == "random":
            should_terms.append(Q("function_score", functions=[{"random_score": {"seed": seed}}]))

        if whitelist is not None and len(whitelist) > 0:

            search = search.query(
                Q(
                    "bool",
                    must=[
                        Q("bool", must=must_terms, should=should_terms, must_not=must_not_terms),
                        Q("ids", type="_doc", values=whitelist),
                    ],
                )
            )

        else:
            search = search.query(Q("bool", must=must_terms, should=should_terms, must_not=must_not_terms))

        if isinstance(sorting, str) and sorting.lower() == "classifier":

            search.sort(
                {
                    "classifier.annotations.value": {
                        "order": "desc",
                        "mode": "sum",
                        "nested": {"path": "classifier", "nested": {"path": "classifier.annotations"}},
                    }
                }
            )

        return search.to_dict()

    def search_db(
        self,
        body,
        size: int = 200,
    ):

        entries = self.database.raw_search(body, size=size)
        return entries

    def entries_lookup(self, entries):
        return self.database.get_entries(entries)

    def __call__(self, query: Dict):
        faulthandler.enable()
        PYTHONFAULTHANDLER = 1

        result = {}
        start_time = time.time()
        logging.info(f"[Searcher] Start searching")

        query = self.parse_query(query)
        logging.info(f"[Searcher] Parsing query done time={time.time()-start_time}")

        query["feature_search"] = self.merge_feature(query["feature_search"])
        logging.info(f"[Searcher] Merged features time={time.time()-start_time}")

        # query = self.parse_query(query)
        # logging.info("Query parsed")

        entries_feature = list(
            self.indexer_plugin_manager.search(
                query["feature_search"],
                collections=query["collections"],
                include_default_collection=query["include_default_collection"],
                size=1000,
            )
        )
        # entries_feature = []
        logging.info(f"[Searcher] Results from indexer len={len(entries_feature)} time={time.time()-start_time}")

        if query["whitelist"] is not None:
            if len(entries_feature) > 0:
                whitelist = list(set(entries_feature) & set(query["whitelist"]))
                if len(whitelist) == 0:
                    whitelist = query["whitelist"]
            else:
                whitelist = query["whitelist"]
        else:
            whitelist = entries_feature

        logging.info(f"whitelist: {whitelist}")

        # logging.info(f"Parsed query: {query}")
        body_old = self.build_search_body_old(
            query["text_search"],
            query["range_search"],
            query["feature_search"],
            whitelist=whitelist,
            sorting=query["sorting"],
            seed=query["seed"],
        )

        # logging.info(f"Parsed query: {query}")
        body = self.build_search_body(
            query["text_search"],
            query["range_search"],
            query["feature_search"],
            whitelist=whitelist,
            sorting=query["sorting"],
            seed=query["seed"],
        )

        # logging.info(f"Old: {body_old}")
        logging.info(f"New: {body}")

        logging.info(f"[Searcher] Start querying database")

        # return []
        entries = self.search_db(body=body, size=max(len(whitelist), 1000))

        entries = list(entries)
        logging.info(f"[Searcher] Database search done len={len(entries)} time={time.time()-start_time}")
        # if len(entries) > 0:
        #     logging.info(entries[0])
        if query["sorting"] == "feature":
            # entries = list(self.mapping_plugin_manager.run(entries, query["feature_search"], ["FeatureL2Mapping"]))
            entries = list(self.mapping_plugin_manager.run(entries, query["feature_search"], ["FeatureCosineMapping"]))

        if query["mapping"] == "umap":
            entries = list(
                self.mapping_plugin_manager.run(
                    entries,
                    query["feature_search"],
                    ["UMapMapping"],
                    configs=[
                        {
                            "type": "UMapMapping",
                            "params": query["mapping_options"],
                        }
                    ],
                )
            )

        logging.info(f"[Searcher] Mapping done len={len(entries)} time={time.time()-start_time}")

        if query["clustering"] == "kmeans":
            entries = list(
                self.mapping_plugin_manager.run(
                    entries,
                    query["feature_search"],
                    ["KMeansMapping"],
                    configs=[
                        {
                            "type": "KMeansMapping",
                            "params": query["clustering_options"],
                        }
                    ],
                )
            )

        logging.info(f"[Searcher] Clustering done len={len(entries)} time={time.time()-start_time}")

        result.update({"entries": list(entries)})

        if self.aggregator and "aggregate" in query:
            if query["aggregate"]["use_query"]:
                aggregations = self.aggregator(
                    query=body["query"], field_names=query["aggregate"]["fields"], size=query["aggregate"]["size"]
                )
            else:
                aggregations = self.aggregator(
                    query=None, field_names=query["aggregate"]["fields"], size=query["aggregate"]["size"]
                )

            result.update({"aggregations": aggregations})

            logging.info(f"[Searcher] Aggregator done len={len(entries)} time={time.time()-start_time}")

        # Clean outputs if not requested
        for entry in entries:
            if "features" not in query["extras"]:
                del entry["feature"]

        logging.info(f"[Searcher] Search done len={len(entries)} time={time.time()-start_time}")
        return result
