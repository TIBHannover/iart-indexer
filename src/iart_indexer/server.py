import logging
from multiprocessing.pool import Pool
import threading
import time
import uuid
import re
import traceback
from concurrent import futures

import copy
import numpy as np
import grpc
from google.protobuf.json_format import MessageToJson, MessageToDict, ParseDict

from iart_indexer import indexer_pb2, indexer_pb2_grpc
from iart_indexer.database.elasticsearch_database import ElasticSearchDatabase
from iart_indexer.database.elasticsearch_suggester import ElasticSearchSuggester
from iart_indexer.plugins import *
from iart_indexer.plugins import ClassifierPlugin, FeaturePlugin, ImageTextPluginManager
from iart_indexer.plugins import IndexerPluginManager
from iart_indexer.utils import image_from_proto, meta_from_proto, meta_to_proto, classifier_to_proto, feature_to_proto
from iart_indexer.utils import get_features_from_db_entry, get_classifier_from_db_entry

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
from iart_indexer.search import Searcher
from iart_indexer.aggregation import Aggregator

from iart_indexer.plugins.cache import Cache

import msgpack

import imageio
from iart_indexer.utils import image_normalize

import spacy

sp_en = spacy.load("en_core_web_sm")
sp_de = spacy.load("de_core_news_sm")


def search(args):
    logging.info("Search: Start")
    try:

        start_time = time.time()
        query = ParseDict(args["query"], indexer_pb2.SearchRequest())
        config = args["config"]

        database = ElasticSearchDatabase(config=config.get("elasticsearch", {}))

        image_text_plugin_manager = globals().get("image_text_manager")
        feature_plugin_manager = globals().get("feature_manager")
        mapping_plugin_manager = globals().get("mapping_manager")
        indexer_plugin_manager = globals().get("indexer_manager")

        classifier_plugin_manager = None
        # indexer_plugin_manager = None

        aggregator = Aggregator(database)
        searcher = Searcher(
            database,
            feature_plugin_manager,
            image_text_plugin_manager,
            classifier_plugin_manager,
            indexer_plugin_manager,
            mapping_plugin_manager,
            aggregator=aggregator,
        )
        logging.info(f"Init done: {time.time()-start_time}")

        search_result = searcher(query)
        logging.info(f"Search done: {time.time()-start_time}")

        result = indexer_pb2.ListSearchResultReply()

        for e in search_result["entries"]:
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

        if "aggregations" in search_result:
            for e in search_result["aggregations"]:
                # logging.info(e)
                aggr = result.aggregate.add()
                aggr.field_name = e["field_name"]
                for y in e["entries"]:
                    value_field = aggr.entries.add()
                    value_field.key = y["name"]
                    value_field.int_val = y["value"]
        result_dict = MessageToDict(result)
        logging.info(result_dict)
        return result_dict
    except Exception as e:
        logging.error(f"Indexer: {repr(e)}")
        # logging.error(traceback.format_exc())

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
    return None


def suggest(args):
    try:
        logging.info("Start suggesting job")
        query = args["query"]
        database = args["database"]

    except Exception as e:
        logging.error(repr(e))


def split_tokens(text):
    return re.findall("[\w]+", text)


def filter_autocompletion(autocompletion_string):
    result_list = [autocompletion_string]
    tokens = split_tokens(autocompletion_string)
    stop_en = sp_en.Defaults.stop_words
    stop_de = sp_de.Defaults.stop_words

    result_list.extend([x for x in tokens if x not in stop_en and x not in stop_de])

    return result_list


def build_suggestion_job(args):

    try:
        database = args["database"]
        suggester = args["suggester"]
        field_names = args["field_names"]

        meta_fields = []
        origin_fields = []
        classifier_fields = []

        for x in field_names:
            split = x.split(".")
            if split[0] == "meta":
                meta_fields.append(split[1])
            if split[0] == "origin":
                origin_fields.append(split[1])
            if split[0] == "classifier":
                classifier_fields.append(split[1])

        db_bulk_cache = []
        for i, x in enumerate(database.all()):
            # print("####################")
            meta_values = []
            if "meta" in x:
                for m in x["meta"]:
                    if m["name"] not in meta_fields:
                        if "*" not in meta_fields:
                            continue

                    if isinstance(m["value_str"], str):
                        meta_values.extend(filter_autocompletion(m["value_str"]))

            origin_values = []
            if "origin" in x:
                for o in x["origin"]:
                    if o["name"] not in origin_fields:
                        if "*" not in origin_fields:
                            continue
                    if isinstance(o["value_str"], str):
                        origin_values.extend(filter_autocompletion(o["value_str"]))

            annotations_values = []
            if "classifier" in x:
                for classifier in x["classifier"]:

                    if classifier["plugin"] not in classifier_fields:
                        if "*" not in classifier_fields:
                            continue

                    for annotations in classifier["annotations"]:
                        annotations_values.extend(filter_autocompletion(annotations["name"]))

            meta_values = list(set(meta_values))
            origin_values = list(set(origin_values))
            annotations_values = list(set(annotations_values))

            db_bulk_cache.append(
                {
                    "id": x["id"],
                    "meta_completion": meta_values,
                    "features_completion": [],
                    "annotations_completion": annotations_values,
                    "origin_completion": origin_values,
                }
            )
            # print(
            #     {
            #         "id": x["id"],
            #         "meta_completion": meta_values,
            #         "features_completion": [],
            #         "annotations_completion": annotations_values,
            #         "origin_completion": origin_values,
            #     }
            # )
            # print()
            # if i > 100:
            #     exit()

            if len(db_bulk_cache) > 1000:

                logging.info(f"BuildSuggester: flush results to database (count:{i} {len(db_bulk_cache)})")
                try_count = 20
                while try_count > 0:
                    try:
                        suggester.bulk_insert(db_bulk_cache)
                        db_bulk_cache = []
                        try_count = 0
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logging.error(f"BuildSuggester: database error (try count: {try_count} {e})")
                        time.sleep(1)
                        try_count -= 1

        if len(db_bulk_cache) > 0:
            logging.info(f"BuildSuggester: flush results to database (count:{i} {len(db_bulk_cache)})")
            suggester.bulk_insert(db_bulk_cache)
    except Exception as e:
        logging.error(f"Indexer: {repr(e)}")
        logging.error(traceback.format_exc())


def build_indexer(args):
    try:
        config = args.get("config")
        database = ElasticSearchDatabase(config=config.get("elasticsearch", {}))
        indexer = globals().get("indexer_manager")

        logging.info(f"Indexer: start")

        class EntryReader:
            def __iter__(self):
                for entry in database.raw_all(
                    {
                        "query": {"function_score": {"random_score": {"seed": 42}}},
                        "size": 250,
                    }
                ):
                    yield get_features_from_db_entry(entry)

        indexer.indexing(EntryReader())

    except Exception as e:
        logging.error(f"Indexer: {repr(e)}")
        logging.error(traceback.format_exc())


def build_feature_cache(args):

    try:
        config = args["config"]
        database = ElasticSearchDatabase(config=config.get("elasticsearch", {}))
        cache = Cache(cache_dir=config.get("cache", {"cache_dir": None})["cache_dir"], mode="a")
        with cache as cache:

            class EntryReader:
                def __iter__(self):
                    for entry in database.all():
                        try:
                            features = get_features_from_db_entry(entry)
                            classifiers = get_classifier_from_db_entry(entry)
                            yield {**features, **classifiers}
                        except Exception as e:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

            for entry in EntryReader():
                cache.write(entry)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def indexing_job(entry):
    try:
        image = imageio.imread(entry["image_data"])
        image = image_normalize(image)
    except Exception as e:
        logging.error(traceback.format_exc())
        return "error", {"id": entry["id"]}
    plugins = []
    for c in entry["cache"]["classifier"]:
        plugins.append({"plugin": c["plugin"], "version": c["version"]})

    classifications = list(indexing_job.classifier_manager.run([image], [plugins]))[0]

    plugins = []
    for c in entry["cache"]["feature"]:
        plugins.append({"plugin": c["plugin"], "version": c["version"]})
    features = list(indexing_job.feature_manager.run([image], [plugins]))[0]

    doc = {"id": entry["id"], "meta": entry["meta"], "origin": entry["origin"]}
    annotations = []
    for f in features["plugins"]:

        for anno in f._annotations:

            for result in anno:
                plugin_annotations = []

                binary_vec = result.feature.binary
                feature_vec = list(result.feature.feature)

                plugin_annotations.append(
                    {
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
    for c in classifications["plugins"]:

        for anno in c._annotations:

            for result in anno:
                plugin_annotations = []
                for concept in result.classifier.concepts:
                    plugin_annotations.append({"name": concept.concept, "type": concept.type, "value": concept.prob})

                classifier_result = {
                    "plugin": result.plugin,
                    "version": result.version,
                    "annotations": plugin_annotations,
                }
                annotations.append(classifier_result)

    if len(annotations) > 0:
        doc["classifier"] = annotations

    # copy predictions from cache
    for exist_c in entry["cache"]["classifier"]:
        if "classifier" not in doc:
            doc["classifier"] = []

        founded = False
        for computed_c in doc["classifier"]:
            if computed_c["plugin"] == exist_c["plugin"] and version.parse(str(computed_c["version"])) > version.parse(
                str(exist_c["version"])
            ):
                founded = True

        if not founded:
            doc["classifier"].append(exist_c)

    for exist_f in entry["cache"]["feature"]:
        if "feature" not in doc:
            doc["feature"] = []
        exist_f_version = version.parse(str(exist_f["version"]))

        founded = False
        for computed_f in doc["feature"]:
            computed_f_version = version.parse(str(computed_f["version"]))
            if computed_f["plugin"] == exist_f["plugin"] and computed_f_version >= exist_f_version:
                founded = True
        if not founded:
            exist_f = {
                "plugin": exist_f["plugin"],
                "version": exist_f["version"],
                "annotations": [
                    {
                        "type": exist_f["type"],
                        "value": exist_f["value"],
                    }
                ],
            }
            doc["feature"].append(exist_f)
    # print(f"FEATURE: {features}")
    # exit()
    # feature_chunk = list(feature_plugin_manager.run([image], filter_feature_plugins))

    # classification_chunk = list(classifier_plugin_manager.run(images_chunk, filter_classifier_plugins))
    # logging.info(doc)
    return "ok", doc


def init_plugins(config):
    data_dict = {}

    image_text_manager = ImageTextPluginManager(configs=config.get("image_text", []))
    image_text_manager.find()

    data_dict["image_text_manager"] = image_text_manager

    feature_manager = FeaturePluginManager(configs=config.get("features", []))
    feature_manager.find()

    data_dict["feature_manager"] = feature_manager

    classifier_manager = ClassifierPluginManager(configs=config.get("classifiers", []))
    classifier_manager.find()

    data_dict["classifier_manager"] = classifier_manager

    indexer_manager = IndexerPluginManager(configs=config.get("indexes", []))
    indexer_manager.find()

    data_dict["indexer_manager"] = indexer_manager

    mapping_manager = MappingPluginManager(configs=config.get("mappings", []))
    mapping_manager.find()

    data_dict["mapping_manager"] = mapping_manager

    cache = Cache(cache_dir=config.get("cache", {"cache_dir": None})["cache_dir"], mode="r")

    data_dict["cache"] = cache

    return data_dict


def init_process(config):
    globals().update(init_plugins(config))


class Commune(indexer_pb2_grpc.IndexerServicer):
    def __init__(self, config):
        self.config = config
        self.managers = init_plugins(config)
        self.process_pool = futures.ProcessPoolExecutor(max_workers=4, initializer=init_process, initargs=(config,))
        self.futures = []

        self.max_results = config.get("indexer", {}).get("max_results", 100)

    def list_plugins(self, request, context):
        reply = indexer_pb2.ListPluginsReply()

        for plugin_name, plugin_class in self.managers["feature_manager"].plugins().items():
            pluginInfo = reply.plugins.add()
            pluginInfo.name = plugin_name
            pluginInfo.type = "feature"

        for plugin_name, plugin_class in self.managers["classifier_manager"].plugins().items():
            pluginInfo = reply.plugins.add()
            pluginInfo.name = plugin_name
            pluginInfo.type = "classifier"

        return reply

    def indexing(self, request_iterator, context):

        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        def filter_and_translate(request_iterator):

            with Cache(cache_dir=self.config.get("cache", {"cache_dir": None})["cache_dir"], mode="r") as cache:
                for x in request_iterator:
                    cache_data = cache[x.image.id]

                    meta = meta_from_proto(x.image.meta)
                    origin = meta_from_proto(x.image.origin)
                    yield {
                        "id": x.image.id,
                        "image_data": x.image.encoded,
                        "meta": meta,
                        "origin": origin,
                        "cache": cache_data,
                    }

        def init_worker(function, config):
            function.feature_manager = FeaturePluginManager(configs=config.get("features", []))
            function.feature_manager.find()

            function.classifier_manager = ClassifierPluginManager(configs=config.get("classifiers", []))
            function.classifier_manager.find()

        with Pool(16, initializer=init_worker, initargs=[indexing_job, self.config]) as p:
            db_bulk_cache = []
            logging.info(f"Indexing: start indexing")
            for i, (status, entry) in enumerate(p.imap(indexing_job, filter_and_translate(request_iterator))):
                if status != "ok":
                    logging.error(f"Indexing: {entry['id']}")
                    yield indexer_pb2.IndexingReply(status="error", id=entry["id"])
                    continue
                # print(entry)
                # logging.info("##########")
                # logging.info(entry)
                db_bulk_cache.append(entry)

                if len(db_bulk_cache) > 1000:

                    logging.info(f"Indexing: flush results to database (count:{i} {len(db_bulk_cache)})")
                    try_count = 20
                    while try_count > 0:
                        try:
                            database.bulk_insert(db_bulk_cache)
                            db_bulk_cache = []
                            try_count = 0
                        except KeyboardInterrupt:
                            raise
                        except:
                            logging.error(f"Indexing: database error (try count: {try_count})")
                            time.sleep(1)
                            try_count -= 1
                # print(entry)
                # print()
                # print(database.get_entry(entry["id"]))
                # exit()

                yield indexer_pb2.IndexingReply(status="ok", id=entry["id"])

            if len(db_bulk_cache) > 0:
                logging.info(f"Indexing: flush results to database (count:{i} {len(db_bulk_cache)})")
                database.bulk_insert(db_bulk_cache)
                db_bulk_cache = []

    def build_suggester(self, request, context):

        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        suggester = ElasticSearchSuggester(config=self.config.get("elasticsearch", {}))
        job_id = uuid.uuid4().hex
        variable = {
            "database": database,
            "suggester": suggester,
            "field_names": list(request.field_names),
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.process_pool.submit(build_suggestion_job, variable)

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

        jsonObj = MessageToDict(request)
        logging.info(jsonObj)

        job_id = uuid.uuid4().hex
        variable = {
            "query": jsonObj,
            "config": self.config,
            "future": None,
            "id": job_id,
        }
        future = self.process_pool.submit(search, copy.deepcopy(variable))
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
            try:
                result = ParseDict(job_data["future"].result(), indexer_pb2.ListSearchResultReply())
                # result = indexer_pb2.ListSearchResultReply.ParseFromString(job_data["future"].result())

            except Exception as e:
                logging.error(f"Indexer: {repr(e)}")
                logging.error(traceback.format_exc())
                result = None
            if result is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Search error")
                return indexer_pb2.ListSearchResultReply()

            return result

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Job unknown")
        return indexer_pb2.ListSearchResultReply()

    def aggregate(self, request, context):

        jsonObj = MessageToJson(request)
        logging.info(jsonObj)
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        aggregator = Aggregator(database)

        size = 5 if request.size <= 0 else request.size
        result_list = []
        if request.part == "meta" and request.type == "count":
            result_list = aggregator.meta_text_count(field_name=request.field_name, size=size)
        elif request.part == "origin" and request.type == "count":
            result_list = aggregator.origin_test_count(field_name=request.field_name, size=size)
        elif request.part == "feature" and request.type == "count":
            result_list = aggregator.feature_count(size=size)
        elif request.part == "classifier" and request.type == "count":
            result_list = aggregator.classifier_tag_count(size=size)
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Job unknown")

        result = indexer_pb2.AggregateReply()
        for x in result_list:
            f = result.field.add()
            f.key = x["name"]
            f.int_val = x["value"]
        return result

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
        logging.info("BUILD_INDEXER")

        job_id = uuid.uuid4().hex
        variable = {
            "config": self.config,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.process_pool.submit(build_indexer, copy.deepcopy(variable))

        for x in range(20):
            print(future)
            time.sleep(0.1)

        variable["future"] = future
        self.futures.append(variable)
        result = indexer_pb2.BuildIndexerReply()

        return result

    def build_feature_cache(self, request, context):
        job_id = uuid.uuid4().hex
        variable = {
            "config": self.config,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.process_pool.submit(build_feature_cache, copy.deepcopy(variable))
        variable["future"] = future
        self.futures.append(variable)
        result = indexer_pb2.BuildFeatureCacheReply()

        return result

    def dump(self, request, context):
        if request.origin is not None and request.origin != "":
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "nested": {
                                    "path": "origin",
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {"match": {f"origin.name": "name"}},
                                                {"match": {f"origin.value_str": request.origin}},
                                            ]
                                        }
                                    },
                                }
                            }
                        ]
                    }
                }
            }
        else:
            body = None
        logging.info(body)
        # return
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        for entry in database.raw_all(body=body):
            yield indexer_pb2.DumpReply(entry=msgpack.packb(entry))

    def load(self, request_iterator, context):
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

        def extract_entry_from_request(request_iterator):
            for r in request_iterator:
                yield msgpack.unpackb(r.entry)

        db_bulk_cache = []
        logging.info(f"Load: start load")
        for i, entry in enumerate(extract_entry_from_request(request_iterator)):
            if entry is None:
                yield indexer_pb2.LoadReply(status="error", id=entry["id"])
                continue
            # print(entry)
            db_bulk_cache.append(entry)

            if len(db_bulk_cache) > 1000:

                logging.info(f"Load: flush results to database (count:{i} {len(db_bulk_cache)})")
                try_count = 20
                while try_count > 0:
                    try:
                        database.bulk_insert(db_bulk_cache)
                        db_bulk_cache = []
                        try_count = 0
                    except KeyboardInterrupt:
                        raise
                    except:
                        logging.error(f"Load: database error (try count: {try_count})")
                        time.sleep(1)
                        try_count -= 1
            # print(entry)
            # print()
            # print(database.get_entry(entry["id"]))
            # exit()

            yield indexer_pb2.LoadReply(status="ok", id=entry["id"])

        if len(db_bulk_cache) > 1:
            logging.info(f"Indexing: flush results to database (count:{i} {len(db_bulk_cache)})")
            database.bulk_insert(db_bulk_cache)
            db_bulk_cache = []


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

        self.commune = Commune(config)

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        indexer_pb2_grpc.add_IndexerServicer_to_server(
            self.commune,
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
                logging.info(
                    f"[Debug] all:{len(self.commune.futures)} done:{len([x for x in self.commune.futures if x['future'].done()])}"
                )
                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)
