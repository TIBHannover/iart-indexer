import os
import re
import math
import uuid
import logging
import msgpack

import faiss

import traceback

import numpy as np
import copy

from datetime import datetime

from iart_indexer.plugins import IndexerPlugin, IndexerPluginManager
from iart_indexer.utils import get_element, get_features_from_db_entry

import time
from iart_indexer.database.elasticsearch_database import ElasticSearchDatabase
from iart_indexer import faiss_indexer_pb2_grpc, faiss_indexer_pb2
import grpc
from concurrent import futures
from threading import Lock

from itertools import zip_longest


@IndexerPluginManager.export("FaissIndexer")
class FaissIndexer(IndexerPlugin):
    default_config = {
        "indexer_dir": "",
        "train_size": 8 * 65536,
        "index_type": "cos",
        "number_of_cluster": 100,
        "one_index_per_collection": True,
        "one_index_for_public": True,
        "number_of_probs": 3,
        "grpc": {"port": 50151, "host": "localhost"}
        # "indexing_size": 105536,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(FaissIndexer, self).__init__(**kwargs)

        self.port = get_element(self.config, "grpc.port")
        if self.port is None:
            self.port = 50151

        self.host = get_element(self.config, "grpc.host")
        if self.host is None:
            self.host = "localhost"
        # Read configs
        logging.info(f"[FaissIndexer] Connection opened")

    def train(self, entries, collections=None):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = faiss_indexer_pb2_grpc.FaissIndexerStub(channel)
        stub.train(faiss_indexer_pb2.TrainRequest(collections=collections))

    def indexing(self, collections=None, rebuild=False):
        logging.info(f"[FaissIndexer] Indexing collections={collections}")

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = faiss_indexer_pb2_grpc.FaissIndexerStub(channel)
        stub.indexing(faiss_indexer_pb2.IndexingRequest(collections=collections))

    def search(self, queries, collections=None, include_default_collection=True, size=100):

        logging.info("KKKK1")
        request = faiss_indexer_pb2.SearchRequest(
            collections=collections,
            include_default_collection=include_default_collection,
        )

        logging.info("KKKK2")
        for q in queries:
            query = request.queries.add()
            query.plugin = q["plugin"]
            query.type = q["type"]
            query.value.extend(q["value"])

        logging.info("KKKK3")
        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        logging.info("KKKK4")
        stub = faiss_indexer_pb2_grpc.FaissIndexerStub(channel)
        logging.info("KKKK5")
        reply = stub.search(request)
        logging.info("KKKK6")
        result = copy.deepcopy(list(reply.ids))

        return result

    def delete(self, collections=None):
        request = faiss_indexer_pb2.DeleteRequest(
            collections=collections,
        )

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        stub = faiss_indexer_pb2_grpc.FaissIndexerStub(channel)
        reply = stub.delete(request)
        return


class FaissEntryReader:
    def __init__(self, database, collections=None, size=None):
        self.database = database
        self.collections = collections
        self.size = size

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
                        "random_score": {"seed": 42},
                    }
                }
            }
        else:
            query = {
                "query": {
                    "function_score": {
                        "query": {
                            "nested": {
                                "path": "collection",
                                "query": {"match": {"collection.is_public": True}},
                            }
                        },
                        "random_score": {"seed": 42},
                    },
                }
            }
        for i, entry in enumerate(self.database.raw_all(query, size=self.size)):
            yield get_features_from_db_entry(entry, return_collection=True)


class Batcher:
    def __init__(self, iterator, size=1):
        self.size = size
        self.iterator = iterator

    def __iter__(self):
        args = [iter(self.iterator)] * self.size
        for x in zip_longest(*args):
            yield x


class FeatureReader:
    def __init__(self, iterator, default_collection_id, batch_size=320):
        self.batch_size = batch_size
        self.iterator = iterator
        self.default_collection_id = default_collection_id

    def __iter__(self):
        iter = Batcher(self.iterator, self.batch_size)

        for batch in iter:
            feature_cache = {}
            for entry in batch:
                if entry is None:
                    continue

                collection_id = get_element(entry, "collection.id")
                is_public = get_element(entry, "collection.is_public")
                if collection_id is None or is_public is None or is_public:
                    collection_id = self.default_collection_id

                for feature in entry["feature"]:

                    index_name = collection_id + "." + feature["plugin"] + "." + feature["type"]
                    if index_name not in feature_cache:
                        feature_cache[index_name] = []
                    feature_cache[index_name].append({"id": entry["id"], "value": feature["value"]})

            results = []
            for k, v in feature_cache.items():
                collection_id, plugin, type = k.split(".")
                results.append({"collection_id": collection_id, "plugin": plugin, "type": type, "entries": v})

            yield results


class FaissIndexerTrainJob:
    @classmethod
    def __call__(cls, args):
        try:

            config = args.get("config")
            faiss_config = args.get("faiss_config")
            indexer = args.get("indexer")
            collections = args.get("collections", None)
            logging.info(f"[FaissIndexerTrainJob] Start train job config={config} config={faiss_config}")
            database = ElasticSearchDatabase(config=config.get("elasticsearch", {}))

            data = {}
            index_names = set()

            logging.info(f"[FaissIndexerTrainJob] Start database reading")

            for i, entry in enumerate(
                FaissEntryReader(database, collections, size=faiss_config.get("train_size", 10000))
            ):
                id = entry["id"]

                for feature in entry["feature"]:
                    index_name = feature["plugin"] + "." + feature["type"]

                    if index_name not in data:
                        data[index_name] = {
                            "entries": {},
                            "data": [],
                            "d": 0,
                            "id": uuid.uuid4().hex,
                            "type": feature["type"],
                            "plugin": feature["plugin"],
                        }
                    data[index_name]["data"].append(feature["value"])
                    data[index_name]["d"] = len(feature["value"])

                    index_names.add(index_name)

            for index_name, index_data in data.items():
                logging.info(f"[FaissIndexerTrainJob] {index_name}:{len(data[index_name]['data'])}")
            logging.info(f"[FaissIndexerTrainJob] Start training of {len(data.items())} indexes")
            id = uuid.uuid4().hex
            collection = {"indexes": [], "id": id, "collection_id": id}
            for index_name, index_data in data.items():
                d = index_data["d"]
                quantizer = faiss.IndexFlatIP(d)
                number_of_clusters = faiss_config.get("number_of_cluster", 100)
                number_of_clusters_to_train = min(number_of_clusters, len(index_data))
                if number_of_clusters_to_train < number_of_clusters:
                    logging.warning(
                        f"[FaissIndexerTrainJob] Dataset size is very small ({number_of_clusters_to_train} < {number_of_clusters})"
                    )

                # Faiss training
                index = faiss.IndexIVFFlat(quantizer, d, number_of_clusters_to_train)
                train_data = np.asarray(index_data["data"]).astype("float32")
                faiss.normalize_L2(train_data)
                index.train(train_data)

                collection["indexes"].append(
                    {
                        "d": index_data["d"],
                        "type": index_data["type"],
                        "plugin": index_data["plugin"],
                        "entries": {},
                        "id": index_data["id"],
                        "index": index,
                    }
                )
            indexer.set_trained_collection(collection)

        except Exception as e:
            logging.error(f"[FaissIndexerTrainJob] {repr(e)}")
            logging.error(traceback.format_exc())


class FaissIndexerIndexingJob:
    @staticmethod
    def __call__(args):
        try:
            config = args.get("config")
            faiss_config = args.get("faiss_config")
            indexer = args.get("indexer")
            collections_to_index = args.get("collections", None)
            logging.info(
                f"[FaissIndexerIndexingJob] Start indexing job config={config} config={faiss_config} collections={collections_to_index}"
            )
            database = ElasticSearchDatabase(config=config.get("elasticsearch", {}))
            print("[FaissIndexerIndexingJob] 1", flush=True)
            # copy all indexes to local variables
            trained_collection = FaissCommune.copy_collection(indexer.trained_collection)
            print("[FaissIndexerIndexingJob] 2", flush=True)
            collections = {k: FaissCommune.copy_collection(v) for k, v in indexer.collections.items()}
            print("[FaissIndexerIndexingJob] 3", flush=True)
            # deafult collection should be part of collections
            default_collection_id = indexer.default_collection

            logging.info(
                f"[FaissIndexerIndexingJob] Found existing default_collection={default_collection_id} collections={[(c['id'], k) for k, c in collections.items()]}"
            )

            # init a new default_collection if no other exists
            if default_collection_id is None:
                default_collection = FaissCommune.copy_collection(trained_collection)
                default_collection_id = default_collection["id"]
                default_collection["collection_id"] = default_collection_id
                collections[default_collection_id] = default_collection
            print("[FaissIndexerIndexingJob] 4", flush=True)
            # only update collections
            collections_to_update = set()
            # rebuild collections
            logging.info(f"[FaissIndexerIndexingJob] Reading features")
            for entries in FeatureReader(FaissEntryReader(database, collections_to_index), default_collection_id):

                logging.info(f"[FaissIndexerIndexingJob] Write features to faiss indexes")
                for entry in entries:

                    collection_id, plugin, type = entry["collection_id"], entry["plugin"], entry["type"]
                    if collection_id not in collections:
                        collections[collection_id] = {
                            **FaissCommune.copy_collection(trained_collection),
                            "collection_id": collection_id,
                            "timestamp": datetime.timestamp(datetime.now()),
                        }

                    for index in collections[collection_id]["indexes"]:
                        if index["type"] == type and index["plugin"] == plugin:
                            ids = [x["id"] for x in entry["entries"] if x["id"] not in index["entries"]]
                            values = [x["value"] for x in entry["entries"] if x["id"] not in index["entries"]]
                            if len(values) == 0:
                                continue
                            values = np.asarray(values).astype("float32")
                            faiss.normalize_L2(values)
                            index["index"].add(values)
                            for id in ids:
                                index["entries"][id] = len(index["entries"])

                            index["rev_entries"] = {v: k for k, v in index["entries"].items()}
                            logging.info(f"Index: {collection_id} {index['type']} {len(index['entries'])}")
                            collections_to_update.add(collection_id)
            print("[FaissIndexerIndexingJob] 5", flush=True)
            new_collections = []
            new_default_collection_id = None
            for k, collection in collections.items():
                if collection["collection_id"] in collections_to_update:
                    renamed_collection = FaissCommune.copy_collection(collection, new_ids=True)
                    new_collections.append(renamed_collection)
                    if collection["id"] == default_collection_id:
                        renamed_collection["collection_id"] = renamed_collection["id"]
                        default_collection_id = renamed_collection["id"]
                        new_default_collection_id = default_collection_id
            indexer.set_collections(new_collections, new_default_collection_id)

        except Exception as e:
            logging.error(f"[FaissIndexerTrainJob] {repr(e)}")
            logging.error(traceback.format_exc())


class FaissCommune(faiss_indexer_pb2_grpc.FaissIndexerServicer):

    default_config = {
        "indexer_dir": "",
        "train_size": 8 * 65536,
        "index_type": "cos",
        "number_of_cluster": 100,
        "one_index_per_collection": True,
        "one_index_for_public": True,
        "number_of_probs": 8,
        # "indexing_size": 105536,
    }

    default_version = 0.1

    def __init__(self, config):
        self.job_pool = futures.ThreadPoolExecutor(max_workers=1)
        self.futures = []
        self.lock = Lock()

        self.faiss_config = self.default_config
        self.config = config

        faiss_indexer_config = None
        indexes = get_element(config, "indexes")

        if isinstance(indexes, (list, set)):
            for x in indexes:
                if x["type"] == "FaissIndexer":
                    faiss_indexer_config = get_element(x, "params")

        if faiss_indexer_config is not None:
            self.faiss_config.update(faiss_indexer_config)

        self.indexer_dir = self.faiss_config["indexer_dir"]
        self.train_size = self.faiss_config["train_size"]
        self.index_type = self.faiss_config["index_type"]
        self.number_of_cluster = self.faiss_config["number_of_cluster"]
        self.one_index_per_collection = self.faiss_config["one_index_per_collection"]
        self.one_index_for_public = self.faiss_config["one_index_for_public"]
        self.number_of_probs = self.faiss_config["number_of_probs"]

        # create some folders
        os.makedirs(self.indexer_dir, exist_ok=True)
        self.collections_dir = os.path.join(self.indexer_dir, "collections")
        os.makedirs(self.collections_dir, exist_ok=True)
        self.indexes_dir = os.path.join(self.indexer_dir, "indexes")
        os.makedirs(self.indexes_dir, exist_ok=True)
        self.collections = {}
        self.default_collection = None
        self.load()

        logging.info(
            f"[FaissIndexer] Latest snapshot loaded ({datetime.fromtimestamp(self.timestamp)}) "
            + f"with {len(self.collections)} collections"
        )

    def load(self):
        indexers = [
            os.path.join(self.indexer_dir, x) for x in os.listdir(self.indexer_dir) if re.match(r"^.*?\.msg$", x)
        ]

        newest_index = {"timestamp": 0.0}

        for x in indexers:
            with open(x, "rb") as f:
                try:
                    data = msgpack.unpackb(f.read(), strict_map_key=False)

                    if newest_index["timestamp"] < data["timestamp"]:
                        newest_index = data
                except:
                    continue

        if "trained_collection" not in newest_index:
            self.timestamp = 0
        else:
            self.trained_collection = self.load_collection(newest_index["trained_collection"])
            self.timestamp = newest_index["timestamp"]
            self.default_collection = newest_index["default_collection"]

            collections_to_delete = []

            for k, v in self.collections:
                if v["id"] not in newest_index["collections"]:
                    collections_to_delete.append(k)

            for x in collections_to_delete:
                del self.collections[k]

            collections_to_load = []

            for collection in newest_index["collections"]:
                if collection not in [x["id"] for x in self.collections.values()]:
                    collections_to_load.append(collection)

            logging.info(f"[FaissIndexer] Loading {collections_to_load} unloading {collections_to_delete}")

            for collection in collections_to_load:
                collection = self.load_collection(collection)
                self.collections[collection["collection_id"]] = collection

    def set_trained_collection(self, collection):
        logging.info(f"[FaissServer] Update trained collection")

        with self.lock:
            self.save_collection(collection)
            self.trained_collection = collection
            self.timestamp = datetime.timestamp(datetime.now())

            with open(os.path.join(self.indexer_dir, uuid.uuid4().hex + ".msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {
                            "collections": [c["id"] for _, c in self.collections.items()],
                            "timestamp": self.timestamp,
                            "default_collection": self.default_collection,
                            "trained_collection": collection["id"],
                        }
                    )
                )

    def set_collections(self, collections, default_collection):
        logging.info(
            f"[FaissServer] Update collections default_collection={default_collection} "
            + f"collections={[{'id':c['id'], 'collection_id':k} for k,c in self.collections.items()]}"
        )

        with self.lock:
            self.timestamp = datetime.timestamp(datetime.now())

            for collection in collections:
                self.save_collection(collection)

            self.collections.update({c["collection_id"]: c for c in collections})

            if default_collection is None:
                default_collection = self.default_collection
            else:
                if self.default_collection is not None:
                    logging.info(f"[FaissServer] Delete old default_collection")
                    del self.collections[self.default_collection]

            self.default_collection = default_collection

            with open(os.path.join(self.indexer_dir, uuid.uuid4().hex + ".msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {
                            "collections": [c["id"] for _, c in self.collections.items()],
                            "timestamp": self.timestamp,
                            "default_collection": default_collection,
                            "trained_collection": self.trained_collection["id"],
                        }
                    )
                )

        logging.info(
            f"[FaissServer] Update collections done default_collection={self.default_collection} "
            + f"collections={[{'id':c['id'], 'collection_id':k} for k,c in self.collections.items()]}"
        )

    def delete_collections(self, collections):
        logging.info(
            f"[FaissServer] Delete collections collections={collections} "
            + f"collections={[{'id':c['id'], 'collection_id':k} for k,c in self.collections.items()]}"
        )

        with self.lock:
            self.timestamp = datetime.timestamp(datetime.now())
            for collection_id in collections:
                if collection_id in self.collections:
                    collection = self.collections[collection_id]
                    del self.collections[collection_id]
                    # delete files from hdd

                    for index in collection["indexes"]:
                        os.remove(os.path.join(self.indexes_dir, index["id"] + ".index"))
                        os.remove(os.path.join(self.indexes_dir, index["id"] + ".msg"))
                    os.remove(os.path.join(self.collections_dir, collection["id"] + ".msg"))

            with open(os.path.join(self.indexer_dir, uuid.uuid4().hex + ".msg"), "wb") as f:
                f.write(
                    msgpack.packb(
                        {
                            "collections": [c["id"] for _, c in self.collections.items()],
                            "timestamp": self.timestamp,
                            "default_collection": self.default_collection,
                            "trained_collection": self.trained_collection["id"],
                        }
                    )
                )
        logging.info(
            f"[FaissServer] Delete collections done default_collection={self.default_collection} "
            + f"collections={[{'id':c['id'], 'collection_id':k} for k,c in self.collections.items()]}"
        )

    @staticmethod
    def copy_collection(collection, new_ids=False):
        print("[copy_collection] ", flush=True)
        new_collection = copy.deepcopy({k: v for k, v in collection.items() if k != "indexes"})

        if "indexes" in collection:
            new_indexes = []

            for index in collection["indexes"]:
                new_indexes.append(FaissCommune.copy_index(index, new_ids=new_ids))

            new_collection["indexes"] = new_indexes

        if new_ids:
            new_collection["id"] = uuid.uuid4().hex

        return new_collection

    @staticmethod
    def copy_index(index, new_ids=False):
        new_index = copy.deepcopy({k: v for k, v in index.items() if k != "index"})
        if "index" in index:
            new_index["index"] = faiss.clone_index(index["index"])
        if new_ids:
            new_index["id"] = uuid.uuid4().hex
        return new_index

    def save_index(self, index):
        faiss.write_index(index["index"], os.path.join(self.indexes_dir, index["id"] + ".index"))
        with open(os.path.join(self.indexes_dir, index["id"] + ".msg"), "wb") as f:
            f.write(msgpack.packb({k: v for k, v in index.items() if k != "index"}))

    def load_index(self, index):
        if isinstance(index, dict):
            index_id = index["id"]
        elif isinstance(index, str):
            index_id = index
        else:
            raise KeyError
        with open(os.path.join(self.indexes_dir, index_id + ".msg"), "rb") as f:
            index = msgpack.unpackb(f.read(), strict_map_key=False)

        index["index"] = faiss.read_index(os.path.join(self.indexes_dir, index_id + ".index"))
        index["index"].nprobe = self.number_of_probs

        # build rev_entries

        index["rev_entries"] = {v: k for k, v in index["entries"].items()}

        return index

    def save_collection(self, collection):
        for index in collection["indexes"]:
            self.save_index(index)

        # delete indexes from collections
        indexes = collection["indexes"]
        collection = {k: v for k, v in collection.items() if k != "indexes"}
        collection["indexes"] = [{"id": i["id"]} for i in indexes]

        with open(os.path.join(self.collections_dir, collection["id"] + ".msg"), "wb") as f:
            f.write(msgpack.packb(collection))

    def load_collection(self, collection):
        if isinstance(collection, dict):
            collection_id = collection["id"]
        elif isinstance(collection, str):
            collection_id = collection
        else:
            raise KeyError

        with open(os.path.join(self.collections_dir, collection_id + ".msg"), "rb") as f:
            collection = msgpack.unpackb(f.read(), strict_map_key=False)

        indexes = []
        for index in collection["indexes"]:
            indexes.append(self.load_index(index))
        collection["indexes"] = indexes

        return collection

    def search(self, request, context):
        logging.info(
            f"[FaissServer] Search collections={request.collections} "
            + f"queries_len={len(request.queries)} "
            + f"include_default_collection={request.include_default_collection}"
        )

        # TODO lock
        ids = []
        collections = list(request.collections)

        if len(collections) == 0:
            collections = [self.default_collection]
        elif request.include_default_collection:
            collections.extend([self.default_collection])

        for collection_id in collections:
            if collection_id not in self.collections:
                logging.warning(f"[FaissServer] Unknown collection {collection_id}")
                continue

            collection = self.collections[collection_id]

            if request.queries:
                for query in request.queries:
                    for index in collection["indexes"]:
                        if index["plugin"] == query.plugin and index["type"] == query.type:
                            feature = np.asarray([list(query.value)]).astype("float32")
                            faiss.normalize_L2(feature)

                            q_result = index["index"].search(feature, k=1000)
                            ids.extend([index["rev_entries"][x.item()] for x in q_result[1][0] if x >= 0])
            else:
                ids.extend(list(collection["indexes"][0]["entries"].keys()))

        return faiss_indexer_pb2.SearchReply(ids=ids)

    def indexing(self, request, context):
        logging.info(f"[FaissServer] Indexing collections={request.collections}")

        collections = list(request.collections)
        if len(collections) == 0:
            collections = None

        job_id = uuid.uuid4().hex
        variable = {
            "config": self.config,
            "faiss_config": self.faiss_config,
            "indexer": self,
            "collections": collections,
            "future": None,
            "id": job_id,
        }
        logging.info(variable)
        future = self.job_pool.submit(FaissIndexerIndexingJob(), variable)

        variable["future"] = future
        self.futures.append(variable)

        return faiss_indexer_pb2.IndexingReply()

    def train(self, request, context):
        logging.info("[FaissServer] Train")

        collections = list(request.collections)
        if len(collections) == 0:
            collections = None

        job_id = uuid.uuid4().hex
        variable = {
            "config": self.config,
            "faiss_config": self.faiss_config,
            "indexer": self,
            "collections": collections,
            "future": None,
            "id": job_id,
        }
        logging.info(variable)
        future = self.job_pool.submit(FaissIndexerTrainJob(), variable)

        variable["future"] = future
        self.futures.append(variable)
        return faiss_indexer_pb2.TrainReply()

    def delete(self, request, context):
        logging.info(f"[FaissServer] Delete {request.collections}")
        self.delete_collections(collections=request.collections)
        return faiss_indexer_pb2.DeleteReply()


class FaissServer:
    def __init__(self, config):
        self.config = config

        # Search port config from plugins
        indexes = get_element(config, "indexes")
        port = None
        if isinstance(indexes, (list, set)):
            for x in indexes:
                if x["type"] == "FaissIndexer":
                    port = get_element(x, "params.grpc.port")

        if port is None:
            logging.error(f"[FaissServer] GRPC port not defined")
            return

        self.commune = FaissCommune(config)

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        faiss_indexer_pb2_grpc.add_FaissIndexerServicer_to_server(
            self.commune,
            self.server,
        )

        self.server.add_insecure_port(f"[::]:{port}")

    def run(self):
        self.server.start()
        logging.info("[FaissServer] Ready")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)
