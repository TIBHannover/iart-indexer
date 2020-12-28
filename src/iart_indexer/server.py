import logging
import threading
import time
import uuid
import traceback
from concurrent import futures

import numpy as np
import grpc
from google.protobuf.json_format import MessageToJson

from iart_indexer import indexer_pb2, indexer_pb2_grpc
from iart_indexer.database.elasticsearch_database import ElasticSearchDatabase
from iart_indexer.database.elasticsearch_suggester import ElasticSearchSuggester
from iart_indexer.plugins import *
from iart_indexer.plugins import ClassifierPlugin, FeaturePlugin
from iart_indexer.plugins import IndexerPluginManager
from iart_indexer.utils import image_from_proto, meta_from_proto, meta_to_proto, classifier_to_proto, feature_to_proto
from iart_indexer.utils import get_features_from_db_entry

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
from iart_indexer.search import Searcher


def indexing(args):
    try:
        # if True:
        logging.info("Start indexing job")
        feature_plugin_manager = args["feature_manager"]
        classifier_plugin_manager = args["classifier_manager"]
        images = args["images"]
        database = args["database"]

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        def prepare_doc():
            plugin_result_list = {}
            for images_chunk in chunks(images, 32):
                feature_chunk = list(feature_plugin_manager.run(images_chunk))

                classification_chunk = list(classifier_plugin_manager.run(images_chunk))

                for img, feature, classification in zip(images_chunk, feature_chunk, classification_chunk):

                    meta = meta_from_proto(img.meta)
                    origin = meta_from_proto(img.origin)
                    doc = {"id": img.id, "meta": meta, "origin": origin}

                    annotations = []
                    for f in feature["plugins"]:

                        for anno in f._annotations:

                            for result in anno:
                                plugin_annotations = []

                                binary_vec = result.feature.binary
                                feature_vec = list(result.feature.feature)

                                hash_splits_list = []
                                for x in range(4):
                                    hash_splits_list.append(
                                        binary_vec[x * len(binary_vec) // 4 : (x + 1) * len(binary_vec) // 4]
                                    )

                                plugin_annotations.append(
                                    {
                                        "hash": {f"split_{i}": x for i, x in enumerate(hash_splits_list)},
                                        "type": result.feature.type,
                                        "value": feature_vec,
                                    }
                                )

                                feature_result = {
                                    "plugin": result.plugin,
                                    "version": result.version,
                                    "annotations": plugin_annotations,
                                }
                                annotations.append(feature_result)

                    if len(annotations) > 0:
                        doc["feature"] = annotations

                    annotations = []
                    for c in classification["plugins"]:

                        for anno in c._annotations:

                            for result in anno:
                                plugin_annotations = []
                                for concept in result.classifier.concepts:
                                    plugin_annotations.append(
                                        {"name": concept.concept, "type": concept.type, "value": concept.prob}
                                    )

                                classifier_result = {
                                    "plugin": result.plugin,
                                    "version": result.version,
                                    "annotations": plugin_annotations,
                                }
                                annotations.append(classifier_result)

                    if len(annotations) > 0:
                        doc["classifier"] = annotations

                    yield doc

        database.bulk_insert(prepare_doc())

        return indexer_pb2.IndexingResult(results=[])

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def search(args):
    query = args["query"]
    feature_plugin_manager = args["feature_manager"]
    mapping_plugin_manager = args["mapping_manager"]
    indexer_plugin_manager = args["indexer_manager"]
    classifier_plugin_manager = None
    # indexer_plugin_manager = None
    database = args["database"]

    searcher = Searcher(
        database, feature_plugin_manager, classifier_plugin_manager, indexer_plugin_manager, mapping_plugin_manager
    )

    entries = searcher(query)

    result = indexer_pb2.ListSearchResultReply()

    for e in entries:
        entry = result.entries.add()
        entry.id = e["id"]
        if "meta" in e:
            meta_to_proto(entry.meta, e["meta"])
        if "origin" in e:
            meta_to_proto(entry.origin, e["origin"])
        if "classifier" in e:
            classifier_to_proto(entry.classifier, e["classifier"])
        if "feature" in e:
            feature_to_proto(entry.feature, e["feature"])
        if "coordinates" in e:
            entry.coordinates.extend(e["coordinates"])

    return result


def suggest(args):
    try:
        # if True:
        logging.info("Start suggesting job")
        query = args["query"]
        database = args["database"]

        #
        # if not request.is_ajax():
        #     return Http404()

    except Exception as e:
        logging.error(repr(e))


def build_autocompletion(args):

    database = args["database"]
    suggester = args["suggester"]
    for x in database.all():
        meta_values = []
        if "meta" in x:
            for key, value in x["meta"].items():
                if key in ["artist_hash"]:
                    continue
                if isinstance(value, str):
                    meta_values.append(value)

        origin_values = []
        if "origin" in x:
            for key, value in x["origin"].items():
                if key in ["link", "license"]:
                    continue
                if isinstance(value, str):
                    origin_values.append(value)

        annotations_values = []
        if "classifier" in x:
            for classifier in x["classifier"]:
                for annotations in classifier["annotations"]:
                    annotations_values.append(annotations["name"])
        suggester.update_entry(hash_id=x["id"], meta=meta_values, annotations=annotations_values)


def build_indexer(args):
    try:
        database = args["database"]
        indexer = args["indexer_manager"]

        class EntryReader:
            def __iter__(self):
                for entry in database.all():
                    yield get_features_from_db_entry(entry)

        indexer.indexing(EntryReader())

    except Exception as e:

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


class Commune(indexer_pb2_grpc.IndexerServicer):
    def __init__(self, config, feature_manager, classifier_manager, indexer_manager, mapping_manager):
        self.config = config
        self.feature_manager = feature_manager
        self.classifier_manager = classifier_manager
        self.indexer_manager = indexer_manager
        self.mapping_manager = mapping_manager
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=8)
        self.futures = []

        self.max_results = config.get("indexer", {}).get("max_results", 100)

    def list_plugins(self, request, context):
        reply = indexer_pb2.ListPluginsReply()

        for plugin_name, plugin_class in self.feature_manager.plugins().items():
            pluginInfo = reply.plugins.add()
            pluginInfo.name = plugin_name
            pluginInfo.type = "feature"

        for plugin_name, plugin_class in self.classifier_manager.plugins().items():
            pluginInfo = reply.plugins.add()
            pluginInfo.name = plugin_name
            pluginInfo.type = "classifier"

        return reply

    def bulk_indexing(self, request, context):
        plugin_list = []
        plugin_name_list = [x.name.lower() for x in request.plugins]
        for plugin_name, plugin_class in self.feature_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue

            plugin_config = {"params": {}}
            for x in self.config["features"]:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)
            plugin_list.append({"plugin": plugin_class, "config": plugin_config})

        for plugin_name, plugin_class in self.classifier_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue
            plugin_config = {"params": {}}
            for x in self.config["classifiers"]:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)
            plugin_list.append({"plugin": plugin_class, "config": plugin_config})
        database = None
        if request.update_database:
            database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        job_id = uuid.uuid4().hex

        variable = {
            "bulk": True,
            "feature_manager": self.feature_manager,
            "classifier_manager": self.classifier_manager,
            "images": request.images,
            "database": database,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }

        future = self.thread_pool.submit(indexing, variable)

        variable["future"] = future
        self.futures.append(variable)

        self.futures = self.futures[-self.max_results :]
        logging.info(f"Cache {len(self.futures)} future references")

        return indexer_pb2.IndexingReply(id=job_id)

    def indexing(self, request, context):
        plugin_list = []
        plugin_name_list = [x.name.lower() for x in request.plugins]
        for plugin_name, plugin_class in self.feature_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue

            plugin_config = {"params": {}}
            for x in self.config["features"]:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)
            plugin_list.append({"plugin": plugin_class, "config": plugin_config})

        for plugin_name, plugin_class in self.classifier_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue
            plugin_config = {"params": {}}
            for x in self.config["classifiers"]:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)
            plugin_list.append({"plugin": plugin_class, "config": plugin_config})
        database = None
        if request.update_database:
            database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        job_id = uuid.uuid4().hex
        variable = {
            "plugin_classes": plugin_list,
            "images": request.images,
            "database": database,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }

        future = self.thread_pool.submit(compute_plugins, variable)

        variable["future"] = future
        self.futures.append(variable)

        self.futures = self.futures[-self.max_results :]
        logging.info(f"Cache {len(self.futures)} future references")

        # thread = threading.Thread(target=compute_plugins, args=(variable,))

        # self.threads[job_id] = (thread, variable)
        # thread.start()

        return indexer_pb2.IndexingReply(id=job_id)

    def build_suggester(self, request, context):

        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        suggester = ElasticSearchSuggester(config=self.config.get("elasticsearch", {}))

        job_id = uuid.uuid4().hex
        variable = {
            "database": database,
            "suggester": suggester,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.thread_pool.submit(build_autocompletion, variable)

        variable["future"] = future
        self.futures.append(variable)

        return indexer_pb2.SuggesterReply(id=job_id)

    def status(self, request, context):

        futures_lut = {x["id"]: i for i, x in enumerate(self.futures)}

        if request.id in futures_lut:
            job_data = self.futures[futures_lut[request.id]]
            done = job_data["future"].done()
            if not done:
                return indexer_pb2.StatusReply(status="running")

            result = job_data["future"].result()
            if result is None:
                return indexer_pb2.StatusReply(status="error")
            return indexer_pb2.StatusReply(status="done", indexing=result)

        return indexer_pb2.StatusReply(status="error")

    def get(self, request, context):

        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        entry = database.get_entry(request.id)
        if entry is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Entry unknown")
            return indexer_pb2.GetReply()

        result = indexer_pb2.GetReply()
        result.id = entry["id"]
        if "meta" in entry:
            meta_to_proto(result.meta, entry["meta"])
        if "origin" in entry:
            meta_to_proto(result.origin, entry["origin"])
        if "classifier" in entry:
            classifier_to_proto(result.classifier, entry["classifier"])
        if "feature" in entry:
            feature_to_proto(result.feature, entry["feature"])
        return result

    def search(self, request, context):

        jsonObj = MessageToJson(request)
        logging.info(jsonObj)
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        job_id = uuid.uuid4().hex
        variable = {
            "feature_manager": self.feature_manager,
            "mapping_manager": self.mapping_manager,
            "indexer_manager": self.indexer_manager,
            "database": database,
            "query": request,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.thread_pool.submit(search, variable)

        variable["future"] = future
        self.futures.append(variable)

        return indexer_pb2.SearchReply(id=job_id)

    def list_search_result(self, request, context):

        futures_lut = {x["id"]: i for i, x in enumerate(self.futures)}

        if request.id in futures_lut:
            job_data = self.futures[futures_lut[request.id]]
            done = job_data["future"].done()
            if not done:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Still running")
                return indexer_pb2.ListSearchResultReply()

            result = job_data["future"].result()
            if result is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Search error")
                return indexer_pb2.ListSearchResultReply()

            return result

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Job unknown")
        return indexer_pb2.ListSearchResultReply()

    def suggest(self, request, context):

        jsonObj = MessageToJson(request)
        logging.info(jsonObj)

        suggester = ElasticSearchSuggester(config=self.config.get("elasticsearch", {}))

        suggestions = suggester.complete(request.query)
        logging.info(suggestions)

        result = indexer_pb2.SuggestReply()
        for group in suggestions:
            if len(group["options"]) < 1:
                continue
            g = result.groups.add()
            g.group = group["type"]
            for suggestion in group["options"]:
                g.suggestions.append(suggestion)

        return result

    def build_indexer(self, request, context):

        jsonObj = MessageToJson(request)
        logging.info(jsonObj)

        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        job_id = uuid.uuid4().hex
        variable = {
            "database": database,
            "feature_manager": self.feature_manager,
            "indexer_manager": self.indexer_manager,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.thread_pool.submit(build_indexer, variable)

        variable["future"] = future
        self.futures.append(variable)
        result = indexer_pb2.IndexerReply()

        return result


def update_database(database, plugin_results):
    for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):

        hash_id = entry.id

        database.update_plugin(
            hash_id,
            plugin_name=plugin_results._plugin.name,
            plugin_version=plugin_results._plugin.version,
            plugin_type=plugin_results._plugin.type,
            annotations=annotations,
        )


class Server:
    def __init__(self, config):
        self.config = config

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        self.feature_manager = FeaturePluginManager(configs=self.config.get("features", []))
        self.feature_manager.find()

        self.classifier_manager = ClassifierPluginManager(configs=self.config.get("classifiers", []))
        self.classifier_manager.find()

        self.indexer_manager = IndexerPluginManager(configs=self.config.get("indexes", []))
        self.indexer_manager.find()

        self.mapping_manager = MappingPluginManager(configs=self.config.get("mappings", []))
        self.mapping_manager.find()

        indexer_pb2_grpc.add_IndexerServicer_to_server(
            Commune(config, self.feature_manager, self.classifier_manager, self.indexer_manager, self.mapping_manager),
            self.server,
        )
        grpc_config = config.get("grpc", {})
        port = grpc_config.get("port", 50051)

        self.server.add_insecure_port(f"[::]:{port}")

    def run(self):
        self.server.start()
        logging.info("Server is now running.")
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            self.server.stop(0)
