import logging
from multiprocessing.pool import Pool
import threading
import time
import uuid
import re
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
from iart_indexer.utils import get_features_from_db_entry, get_classifier_from_db_entry

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
from iart_indexer.search import Searcher

from iart_indexer.plugins.cache import Cache

import msgpack

import imageio
from iart_indexer.utils import image_normalize

import spacy

sp_en = spacy.load("en_core_web_sm")
sp_de = spacy.load("de_core_news_sm")


def indexing(args):
    # try:
    if True:
        logging.info("Start indexing job")
        feature_plugin_manager = args["feature_manager"]
        classifier_plugin_manager = args["classifier_manager"]
        images = args["images"]
        database = args["database"]
        cache = args["cache"]
        existing_predictions = {}
        filter_feature_plugins = []
        filter_classifier_plugins = []

        with cache as cache:
            for image in images:
                cache_data = cache[image.id]
                existing_predictions[image.id] = cache_data

                plugins = []
                for c in cache_data["classifier"]:
                    plugins.append({"plugin": c["plugin"], "version": c["version"]})
                filter_classifier_plugins.append(plugins)

                plugins = []
                for c in cache_data["feature"]:
                    plugins.append({"plugin": c["plugin"], "version": c["version"]})
                filter_feature_plugins.append(plugins)

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        def prepare_doc():
            for images_chunk in chunks(images, 32):

                feature_chunk = list(feature_plugin_manager.run(images_chunk, filter_feature_plugins))

                classification_chunk = list(classifier_plugin_manager.run(images_chunk, filter_classifier_plugins))

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

                    # copy predictions from cache
                    for exist_c in existing_predictions[img.id]["classifier"]:
                        if "classifier" not in doc:
                            doc["classifier"] = []

                        founded = False
                        for computed_c in doc["classifier"]:
                            if computed_c["plugin"] == exist_c["plugin"] and version.parse(
                                str(computed_c["version"])
                            ) > version.parse(str(exist_c["version"])):
                                founded = True

                        if not founded:
                            doc["classifier"].append(exist_c)

                    for exist_f in existing_predictions[img.id]["feature"]:
                        if "feature" not in doc:
                            doc["feature"] = []
                        exist_f_version = version.parse(str(exist_f["version"]))

                        founded = False
                        for computed_f in doc["feature"]:
                            computed_f_version = version.parse(str(computed_f["version"]))
                            if computed_f["plugin"] == exist_f["plugin"] and computed_f_version >= exist_f_version:
                                founded = True
                        if not founded:
                            # TODO build hash
                            output = np.asarray(exist_f["value"])
                            output_bin = (output > 0).astype(np.int32).tolist()
                            output_hash = "".join([str(x) for x in output_bin])
                            hash_splits_list = []
                            for x in range(4):
                                hash_splits_list.append(
                                    output_hash[x * len(output_hash) // 4 : (x + 1) * len(output_hash) // 4]
                                )
                            exist_f = {
                                "plugin": exist_f["plugin"],
                                "version": exist_f["version"],
                                "annotations": [
                                    {
                                        "type": exist_f["type"],
                                        "hash": {f"split_{i}": x for i, x in enumerate(hash_splits_list)},
                                        "value": exist_f["value"],
                                    }
                                ],
                            }
                            doc["feature"].append(exist_f)

                    yield doc

        database.bulk_insert(prepare_doc())

        return indexer_pb2.IndexingResult(results=[])

    # except Exception as e:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def search(args):
    start_time = time.time()
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
    logging.info(f"Init done: {time.time()-start_time}")

    entries = searcher(query)
    logging.info(f"Search done: {time.time()-start_time}")

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


def split_tokens(text):
    return re.findall("[\w]+", text)


def filter_autocompletion(autocompletion_string):
    result_list = [autocompletion_string]
    tokens = split_tokens(autocompletion_string)
    stop_en = sp_en.Defaults.stop_words
    stop_de = sp_de.Defaults.stop_words

    result_list.extend([x for x in tokens if x not in stop_en and x not in stop_de])

    return result_list


def build_autocompletion(args):

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
                    if m["name"] not in origin_fields:
                        if "*" not in origin_fields:
                            continue
                    if isinstance(m["value_str"], str):
                        origin_values.extend(filter_autocompletion(m["value_str"]))

            annotations_values = []
            if "classifier" in x:
                for classifier in x["classifier"]:

                    if classifier["plugin"] not in classifier_fields:
                        if "*" not in classifier_fields:
                            continue

                    for annotations in classifier["annotations"]:
                        annotations_values.extend(filter_autocompletion(m["name"]))

            meta_values = list(set(meta_values))
            origin_values = list(set(origin_values))
            annotations_values = list(set(annotations_values))

            suggester.update_entry(
                hash_id=x["id"], meta=meta_values, annotations=annotations_values, origins=origin_values
            )

            if i % 10000 == 0:
                logging.info(f"build_autocompletion: {i}")

    except Exception as e:

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
        logging.error(repr(e))


def build_indexer(args):
    try:
        database = args["database"]
        indexer = args["indexer_manager"]

        logging.info(f"Indexer: start")

        class EntryReader:
            def __iter__(self):
                for entry in database.all():
                    yield get_features_from_db_entry(entry)

        indexer.indexing(EntryReader())

    except Exception as e:
        logging.info(f"Indexer: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def build_feature_cache(args):
    try:
        database = args["database"]
        cache = args["cache"]
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

                            # print(entry)
                            # print(features)
                            # print(classifiers)
                            # exit()

            for entry in EntryReader():
                cache.write(entry)

    except Exception as e:

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def indexing_job(entry):

    image = imageio.imread(entry["image_data"])
    image = image_normalize(image)

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

                hash_splits_list = []
                for x in range(4):
                    hash_splits_list.append(binary_vec[x * len(binary_vec) // 4 : (x + 1) * len(binary_vec) // 4])

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
            # TODO build hash
            output = np.asarray(exist_f["value"])
            output_bin = (output > 0).astype(np.int32).tolist()
            output_hash = "".join([str(x) for x in output_bin])
            hash_splits_list = []
            for x in range(4):
                hash_splits_list.append(output_hash[x * len(output_hash) // 4 : (x + 1) * len(output_hash) // 4])
            exist_f = {
                "plugin": exist_f["plugin"],
                "version": exist_f["version"],
                "annotations": [
                    {
                        "type": exist_f["type"],
                        "hash": {f"split_{i}": x for i, x in enumerate(hash_splits_list)},
                        "value": exist_f["value"],
                    }
                ],
            }
            doc["feature"].append(exist_f)
    # print(f"FEATURE: {features}")
    # exit()
    # feature_chunk = list(feature_plugin_manager.run([image], filter_feature_plugins))

    # classification_chunk = list(classifier_plugin_manager.run(images_chunk, filter_classifier_plugins))

    return doc


class Commune(indexer_pb2_grpc.IndexerServicer):
    def __init__(self, config, feature_manager, classifier_manager, indexer_manager, mapping_manager, cache):
        self.config = config
        self.feature_manager = feature_manager
        self.classifier_manager = classifier_manager
        self.indexer_manager = indexer_manager
        self.mapping_manager = mapping_manager
        self.cache = cache
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=16)
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
            "cache": self.cache,
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
            for i, entry in enumerate(p.imap(indexing_job, filter_and_translate(request_iterator))):
                if entry is None:
                    yield indexer_pb2.IndexingReply(status="error", id=entry["id"])
                    continue
                # print(entry)
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

            if len(db_bulk_cache) > 1:
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
        print("WTF")
        logging.info("BUILD_INDEXER")
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
        result = indexer_pb2.BuildIndexerReply()

        return result

    def build_feature_cache(self, request, context):
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        job_id = uuid.uuid4().hex
        variable = {
            "database": database,
            "cache": Cache(cache_dir=self.config.get("cache", {"cache_dir": None})["cache_dir"], mode="a"),
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
            "id": job_id,
        }
        future = self.thread_pool.submit(build_feature_cache, variable)

        variable["future"] = future
        self.futures.append(variable)
        result = indexer_pb2.BuildFeatureCacheReply()

        return result

    def dump(self, request, context):
        database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))
        for entry in database.all():
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

        self.cache = Cache(cache_dir=self.config.get("cache", {"cache_dir": None})["cache_dir"], mode="r")

        indexer_pb2_grpc.add_IndexerServicer_to_server(
            Commune(
                config,
                self.feature_manager,
                self.classifier_manager,
                self.indexer_manager,
                self.mapping_manager,
                self.cache,
            ),
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
