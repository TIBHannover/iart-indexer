import logging
import uuid
import json
import indexer_pb2
from typing import Dict, List
from iart_indexer.utils import get_features_from_db_entry


class Searcher:
    def __init__(
        self,
        database,
        feature_plugin_manager=None,
        classifier_plugin_manager=None,
        indexer_plugin_manager=None,
        mapping_plugin_manager=None,
    ):
        super().__init__()
        self.database = database
        self.feature_plugin_manager = feature_plugin_manager
        self.classifier_plugin_manager = classifier_plugin_manager
        self.indexer_plugin_manager = indexer_plugin_manager
        self.mapping_plugin_manager = mapping_plugin_manager

    def parse_query(self, query):
        logging.info(query)
        text_search = []
        feature_search = []
        # search_terms = []
        for term in query.terms:

            term_type = term.WhichOneof("term")
            if term_type == "text":
                text = term.text
                field = text.field.lower()
                logging.info(text.flag)
                if text.flag == indexer_pb2.TextSearchTerm.MUST:
                    flag = "must"
                if text.flag == indexer_pb2.TextSearchTerm.SHOULD:
                    flag = "should"

                text_search.append({"field": field, "query": text.query, "flag": flag})

            if term_type == "feature":
                feature = term.feature

                query_feature = []

                entry = None
                if feature.image.id is not None and feature.image.id != "":
                    entry = self.database.get_entry(feature.image.id)

                if entry is not None:
                    entry = get_features_from_db_entry(entry)

                    for p in feature.plugins:
                        # TODO add weight
                        for f in entry["feature"]:

                            if p.name.lower() == f["plugin"].lower():

                                feature_search.append(
                                    {
                                        **f,
                                        "weight": p.weight,
                                    }
                                )

                # TODO add example search here
                else:
                    logging.info("Feature")

                    feature_results = list(
                        self.feature_plugin_manager.run(
                            [feature.image]
                        )  # , plugins=[x.name.lower() for x in feature.plugins])
                    )[0]

                    for p in feature.plugins:
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
                                                "weight": p.weight,
                                            }
                                        )

        return text_search, feature_search, query.sorting

    def build_search_body(
        self,
        text_search: List = [],
        feature_search: List = [],
        whitelist: List = [],
        sorting=None,
        size: int = 200,
    ):

        must_terms = []
        should_terms = []
        for e in text_search:
            term = None
            if "field" not in e or e["field"] is None:
                term = {
                    "multi_match": {
                        "query": e["query"],
                        "fields": ["meta_text", "classifier_text"],
                    }
                }
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
            search_terms.append({"function_score": {"functions": [{"random_score": {"seed": uuid.uuid4().hex}}]}})

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
        logging.info("Start searching")
        text_search, feature_search, sorting = self.parse_query(query)

        logging.info("Query parsed")
        entries_feature = self.indexer_plugin_manager.search(feature_search, size=1000)
        if len(entries_feature) > 0:
            resutl = list(self.entries_lookup(entries_feature))
            # print(resutl[:10])
        logging.info(f"Parsed query: {text_search} {feature_search} {sorting}")
        body = self.build_search_body(
            text_search,
            feature_search,
            whitelist=entries_feature,
            sorting=sorting,
        )

        logging.info(json.dumps(body, indent=2))
        # return []
        entries = self.search_db(body=body, size=max(len(entries_feature), 100))
        entries = list(entries)
        # if len(entries) > 0:
        #     logging.info(entries[0])
        logging.info(f"Entries 1 {len(entries)}")
        # return []
        if query.sorting.lower() == "feature":
            entries = list(self.mapping_plugin_manager.run(entries, feature_search, ["FeatureL2Mapping"]))

        logging.info(f"Entries 2 {len(entries)}")
        if query.mapping.lower() == "umap":
            entries = list(self.mapping_plugin_manager.run(entries, feature_search, ["UMapMapping"]))

        logging.info(f"Entries 3 {len(entries)}")
        return list(entries)[:100]
