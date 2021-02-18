import logging
import uuid
import json
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

        self.meta_lut = {
            "title": "meta.title",
            "author": "meta.author",
            "location": "meta.location",
            "institution": "meta.institution",
        }

    def parse_query(self, query):
        text_search = []
        classifier_search = []
        feature_search = []
        # search_terms = []
        for term in query.terms:

            term_type = term.WhichOneof("term")
            if term_type == "meta":
                meta = term.meta
                if meta.field is not None and meta.field.lower() in self.meta_lut:
                    field = self.meta_lut[term.field.lower()]
                else:
                    field = None

                text_search.append({"field": field, "query": meta.query})

            if term_type == "classifier":
                classifier = term.classifier
                # TODO add lut
                classifier_search.append({"query": classifier.query})

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

        return text_search, classifier_search, feature_search, query.sorting

    def search_db(
        self,
        text_search: List = [],
        classifier_search: List = [],
        feature_search: List = [],
        whitelist: List = [],
        sorting=None,
        size: int = 200,
    ):

        search_terms = []
        logging.info(text_search)
        for e in text_search:
            if e["field"] is not None:
                fields = [e["field"]]
                search_terms.append(
                    {
                        "multi_match": {
                            "query": e["query"],
                            "fields": fields,
                        }
                    }
                )
            else:
                search_terms.append(
                    {
                        "multi_match": {
                            "query": e["query"],
                            "fields": "meta_text",
                        }
                    }
                )

        logging.info(search_terms)

        for e in classifier_search:
            search_terms.append(
                {
                    "nested": {
                        "path": "classifier",
                        "query": {
                            "nested": {
                                "path": "classifier.annotations",
                                "query": {"bool": {"should": [{"match": {"classifier.annotations.name": e["query"]}}]}},
                            }
                        },
                    }
                }
            )

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
                        "must": [{"ids": {"type": "_doc", "values": whitelist}}, {"bool": {"should": search_terms}}]
                    }
                }
            }

        else:
            body = {"query": {"bool": {"should": search_terms}}}

        if elastic_sorting is not None:
            body.update({"sort": elastic_sorting})

        logging.info(json.dumps(body, indent=2))
        entries = self.database.raw_search(body, size=size)
        return entries

    def entries_lookup(self, entries):
        return self.database.get_entries(entries)

    def __call__(self, query: Dict):
        logging.info("Start searching")
        text_search, classifier_search, feature_search, sorting = self.parse_query(query)

        entries_feature = self.indexer_plugin_manager.search(feature_search, size=1000)
        if len(entries_feature) > 0:
            resutl = list(self.entries_lookup(entries_feature))
            # print(resutl[:10])
        logging.info(f"Parsed query: {text_search} {classifier_search} {feature_search} {sorting}")
        entries = self.search_db(
            text_search,
            classifier_search,
            feature_search,
            whitelist=entries_feature,
            sorting=sorting,
            size=max(len(entries_feature), 100),
        )

        if query.sorting.lower() == "feature":
            entries = list(self.mapping_plugin_manager.run(entries, feature_search, ["FeatureCosineMapping"]))

        if query.mapping.lower() == "umap":
            entries = list(self.mapping_plugin_manager.run(entries, feature_search, ["UMapMapping"]))

        return list(entries)[:100]
