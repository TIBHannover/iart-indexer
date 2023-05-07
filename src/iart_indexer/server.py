import re
import copy
import grpc
import time
import uuid
# import spacy
import msgpack
import imageio
import logging
import traceback
import threading

import numpy as np

from concurrent import futures
from multiprocessing.pool import Pool
from google.protobuf.json_format import MessageToJson, MessageToDict, ParseDict

from iart_indexer import indexer_pb2, indexer_pb2_grpc
from iart_indexer.database.elasticsearch_database import ElasticSearchDatabase
from iart_indexer.database.elasticsearch_suggester import ElasticSearchSuggester
from iart_indexer.plugins import *
from iart_indexer.plugins import ClassifierPlugin, FeaturePlugin, ImageTextPluginManager
from iart_indexer.plugins import IndexerPluginManager
from iart_indexer.utils import image_from_proto, meta_from_proto, meta_to_proto, classifier_to_proto, feature_to_proto
from iart_indexer.utils import get_features_from_db_entry, get_classifier_from_db_entry, read_chunk, image_normalize
from iart_indexer.jobs import IndexingJob
from iart_indexer.search import Searcher
from iart_indexer.aggregation import Aggregator
from iart_indexer.plugins.cache import Cache

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# sp_en = spacy.load("en_core_web_sm")
# sp_de = spacy.load("de_core_news_sm")


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

        logging.info(f"Init done: {time.time() - start_time}")

        search_result = searcher(query)
        logging.info(f"Search done: {time.time() - start_time}")

        result = indexer_pb2.ListSearchResultReply()

        for e in search_result["entries"]:
            entry = result.entries.add()
            entry.id = e["id"]
            entry.padded = e["padded"]

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
            if "distance" in e:
                entry.distance = e["distance"]
            if "cluster" in e:
                entry.cluster = e["cluster"]
            if "collection" in e:
                entry.collection.id = e["collection"]["id"]
                entry.collection.name = e["collection"]["name"]
                entry.collection.is_public = e["collection"]["is_public"]

        if "aggregations" in search_result:
            for e in search_result["aggregations"]:
                aggr = result.aggregate.add()
                aggr.field_name = e["field_name"]

                for y in e["entries"]:
                    value_field = aggr.entries.add()
                    value_field.key = y["name"]
                    value_field.int_val = y["value"]

        result_dict = MessageToDict(result)

        return result_dict
    except Exception as e:
        logging.error(f"Indexer: {repr(e)}")
        exc_type, exc_value, exc_traceback = sys.exc_info()

        traceback.print_exception(
            exc_type,
            exc_value,
            exc_traceback,
            limit=2,
            file=sys.stdout,
        )

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
    logging.info(f"[IndexerJob] Starting")

    try:
        config = args.get("config")
        rebuild = args.get("rebuild")
        collections = args.get("collections")

        database = ElasticSearchDatabase(config=config.get("elasticsearch", {}))
        if collections is None or len(collections) <= 0:
            collections = [
                x["key"]
                for x in database.raw_aggregate(
                    {
                        "aggs": {
                            "c": {
                                "nested": {"path": "collection"},
                                "aggs": {"c": {"terms": {"field": "collection.id.keyword"}}},
                            }
                        }
                    }
                )["aggregations"]["c"]["c"]["buckets"]
            ]

            # collections =
        logging.info(f"[IndexerJob] Start parameters rebuild={rebuild} collections={collections}")
        indexer = globals().get("indexer_manager")

        class EntryReader:
            def __init__(self, collections=None):
                self.collections = collections

            def __iter__(self):
                if self.collections is not None:
                    query = {
                        "query": {
                            "function_score": {
                                "query": {
                                    "nested": {
                                        "path": "collection",
                                        "query": {"terms": {"collection.id": self.collections}},
                                    }
                                },
                                "random_score": {
                                    "seed": 42,
                                },
                            }
                        },
                        "size": 1000,
                    }
                else:
                    query = {
                        "query": {
                            "function_score": {
                                "random_score": {
                                    "seed": 42,
                                },
                            }
                        },
                        "size": 1000,
                    }

                for entry in database.raw_all(query):
                    yield get_features_from_db_entry(entry, return_collection=True)

        # for e in EntryReader(collections):
        #     print(e)
        # print(len(list(EntryReader(collections))))
        # return

        indexer.indexing(
            index_entries=EntryReader(collections),
            collections=collections,
            rebuild=rebuild,
        )

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
        self.process_pool = futures.ProcessPoolExecutor(max_workers=1, initializer=init_process, initargs=(config,))
        self.indexing_process_pool = futures.ProcessPoolExecutor(
            max_workers=8, initializer=IndexingJob().init_worker, initargs=(config,)
        )
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

        def filter_and_translate(cache, request_iterator):
            logging.info("[Server]: Start reading cache for indexing")

            for x in request_iterator:
                cache_data = cache[x.image.id]

                meta = meta_from_proto(x.image.meta)
                origin = meta_from_proto(x.image.origin)

                collection = {}

                if x.image.collection.id != "":
                    collection["id"] = x.image.collection.id
                    collection["name"] = x.image.collection.name
                    collection["is_public"] = x.image.collection.is_public

                yield {
                    "id": x.image.id,
                    "image_data": x.image.encoded,
                    "meta": meta,
                    "origin": origin,
                    "collection": collection,
                    "cache": cache_data,
                }

        db_bulk_cache = []
        logging.info(f"[Server] Indexing: start indexing")

        request_iter = iter(request_iterator)
        collections = set()

        with Cache(cache_dir=self.config.get("cache", {"cache_dir": None})["cache_dir"], mode="r") as cache:
            while True:
                chunk = read_chunk(filter_and_translate(cache, request_iter), chunksize=512)
                if len(chunk) <= 0:
                    break

                for i, (status, entry) in enumerate(self.indexing_process_pool.map(IndexingJob(), chunk, chunksize=64)):
                    if status != "ok":
                        logging.error(f"Indexing: {entry['id']}")
                        yield indexer_pb2.IndexingReply(status="error", id=entry["id"])
                        continue

                    collection = entry.get("collection")

                    if collection:
                        collection_id = collection.get("id")

                        if collection_id:
                            collections.add(collection_id)

                    db_bulk_cache.append(entry)

                    if len(db_bulk_cache) > 64:
                        logging.info(f"[Server] Indexing: flush results to database (count:{i} {len(db_bulk_cache)})")
                        try_count = 20

                        while try_count > 0:
                            try:
                                database.bulk_insert(db_bulk_cache)
                                db_bulk_cache = []
                                try_count = 0
                            except KeyboardInterrupt:
                                raise
                            except:
                                logging.error(f"[Server] Indexing: database error (try count: {try_count})")
                                time.sleep(1)
                                try_count -= 1

                    yield indexer_pb2.IndexingReply(status="ok", id=entry["id"])

        if len(db_bulk_cache) > 0:
            logging.info(f"[Server] Indexing flush results to database (count:{i} {len(db_bulk_cache)})")
            database.bulk_insert(db_bulk_cache)
            db_bulk_cache = []

        # start indexing all
        self.managers["indexer_manager"].indexing(collections)

    def build_suggester(self, request, context):
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        suggester = ElasticSearchSuggester(config=self.config.get("elasticsearch", {}))
        job_id = uuid.uuid4().hex

        variable = {
            "database": database,
            "suggester": suggester,
            "field_names": list(request.field_names),
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
        if "collection" in entry:
            logging.info(entry["collection"])
            result.collection.id = entry["collection"]["id"]
            result.collection.name = entry["collection"]["name"]
            result.collection.is_public = entry["collection"]["is_public"]
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
                result = job_data["future"].result()
                result = ParseDict(result, indexer_pb2.ListSearchResultReply())
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

        logging.info(f"rebuild {request.rebuild} {request.collections}")
        job_id = uuid.uuid4().hex
        variable = {
            "config": self.config,
            "rebuild": request.rebuild,
            "collections": list(request.collections),
            "future": None,
            "id": job_id,
        }
        logging.info(variable)
        future = self.process_pool.submit(build_indexer, copy.deepcopy(variable))

        variable["future"] = future
        self.futures.append(variable)
        result = indexer_pb2.BuildIndexerReply()

        return result

    def build_feature_cache(self, request, context):
        job_id = uuid.uuid4().hex
        variable = {
            "config": self.config,
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

            yield indexer_pb2.LoadReply(status="ok", id=entry["id"])

        if len(db_bulk_cache) > 1:
            logging.info(f"Indexing: flush results to database (count:{i} {len(db_bulk_cache)})")
            database.bulk_insert(db_bulk_cache)
            db_bulk_cache = []

    def collection_delete(self, request, context):
        logging.info("[Server] collection_delete")
        logging.info(request.id)

        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        result = database.raw_delete(
            {
                "query": {
                    "nested": {
                        "path": "collection",
                        "query": {"terms": {"collection.id": [request.id]}},
                    }
                }
            }
        )

        logging.info(f"[Server] {result}")

        result = indexer_pb2.CollectionDeleteReply()  # collections,ids)

        # start indexing all
        self.managers["indexer_manager"].delete([request.id])

        return result


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
        logging.info("[Server] Ready")

        try:
            while True:
                num_jobs = len(self.commune.futures)
                num_jobs_done = len([x for x in self.commune.futures if x["future"].done()])
                logging.info(f"[Server] num_jobs:{num_jobs} num_jobs_done:{num_jobs_done}")

                time.sleep(60*60)
        except KeyboardInterrupt:
            self.server.stop(0)
