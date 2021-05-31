import os
import re
import math
import uuid
import logging
import msgpack

try:
    import faiss
except ImportError as error:
    pass

import numpy as np
import copy

from datetime import datetime

from iart_indexer import indexer_pb2
from iart_indexer.plugins import IndexerPlugin, IndexerPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize, get_element


@IndexerPluginManager.export("FaissIndexer")
class FaissIndexer(IndexerPlugin):
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

    def __init__(self, **kwargs):
        super(FaissIndexer, self).__init__(**kwargs)

        # Read configs
        self.indexer_dir = self.config["indexer_dir"]
        self.train_size = self.config["train_size"]
        self.index_type = self.config["index_type"]
        self.number_of_cluster = self.config["number_of_cluster"]
        self.one_index_per_collection = self.config["one_index_per_collection"]
        self.one_index_for_public = self.config["one_index_for_public"]
        self.number_of_probs = self.config["number_of_probs"]
        # self.indexing_size = self.config["indexing_size"]

        # create some folders
        os.makedirs(self.indexer_dir, exist_ok=True)
        self.collections_dir = os.path.join(self.indexer_dir, "collections")
        os.makedirs(self.collections_dir, exist_ok=True)
        self.indexes_dir = os.path.join(self.indexer_dir, "indexes")
        os.makedirs(self.indexes_dir, exist_ok=True)

        self.load()

    def check_changes(self):
        pass

    def load(self):
        indexers = [
            os.path.join(self.indexer_dir, x) for x in os.listdir(self.indexer_dir) if re.match(r"^.*?\.msg$", x)
        ]
        logging.info(indexers)
        logging.info(os.listdir(self.indexer_dir))
        newest_index = {"timestamp": 0.0}
        for x in indexers:
            logging.info(x)
            with open(x, "rb") as f:
                data = msgpack.unpackb(f.read())
                if newest_index["timestamp"] < data["timestamp"]:
                    newest_index = data

        logging.info(newest_index)
        logging.info("####################")
        if "trained_collection" not in newest_index:
            self.not_init = True
        else:
            self.trained_collection = self.load_collection(newest_index["trained_collection"])
            self.default_collection = self.load_collection(newest_index["default_collection"])
            self.collections = {}
            for collection in newest_index["collections"]:
                collection = self.load_collection(collection)
                self.collections[collection["collection_id"]] = collection
            self.not_init = False

    @staticmethod
    def copy_collection(collection, new_ids=True):
        new_collection = copy.deepcopy({k: v for k, v in collection.items() if k != "indexes"})
        if "indexes" in collection:
            new_indexes = []
            for index in collection["indexes"]:
                new_indexes.append(FaissIndexer.copy_index(index, new_ids=new_ids))
            new_collection["indexes"] = new_indexes

        if new_ids:
            new_collection["id"] = uuid.uuid4().hex

        return new_collection

    @staticmethod
    def copy_index(index, new_ids=True):
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
            index = msgpack.unpackb(f.read())

        index["index"] = faiss.read_index(os.path.join(self.indexes_dir, index_id + ".index"))
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
            collection = msgpack.unpackb(f.read())

        indexes = []
        for index in collection["indexes"]:
            indexes.append(self.load_index(index))
        collection["indexes"] = indexes

        return collection

    def train(self, entries):
        data = {}
        index_names = set()
        for i, entry in enumerate(entries):
            id = entry["id"]

            for feature in entry["feature"]:
                index_name = feature["plugin"] + "." + feature["type"]

                if index_name not in data:
                    data[index_name] = {
                        "entries": [],
                        "data": [],
                        "d": 0,
                        "id": uuid.uuid4().hex,
                        "type": feature["type"],
                        "plugin": feature["plugin"],
                    }
                data[index_name]["data"].append(feature["value"])
                data[index_name]["d"] = len(feature["value"])

                index_names.add(index_name)

            if i % 1000 == 0 and i > 0:
                logging.info(f"[FaissIndexer] Read {i}")

                for index_name, index_data in data.items():
                    logging.info(f"[FaissIndexer] {index_name}:{len(data[index_name]['data'])}")
            if i > self.train_size:
                break
                # break

        logging.info(f"[FaissIndexer] Start training")
        collection = {"indexes": []}
        for index_name, index_data in data.items():
            d = index_data["d"]
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.number_of_cluster)

            train_data = np.asarray(index_data["data"]).astype("float32")
            faiss.normalize_L2(train_data)
            index.train(train_data)
            # index.add(train_data)

            # faiss.write_index(index, os.path.join(self.indexer_dir, index_data["id"] + ".index"))

            collection["indexes"].append(
                {
                    "d": index_data["d"],
                    "type": index_data["type"],
                    "plugin": index_data["plugin"],
                    "entries": [],
                    "id": index_data["id"],
                    "index": index,
                }
            )
        return collection

    def indexing(self, train_entries, index_entries, rebuild=False):
        logging.info(f"[FaissIndexer] Indexing (rebuild={rebuild})")

        new_trained_collection = False
        if rebuild:
            trained_collection = FaissIndexer.copy_collection(self.train(train_entries))
            trained_collection_id = trained_collection["id"]
            collections = {}
            # deafult collection should be part of collections
            default_collection = FaissIndexer.copy_collection(trained_collection)
            default_collection_id = default_collection["id"]
            default_collection["collection_id"] = default_collection_id
            new_trained_collection = True
            collections[default_collection_id] = default_collection
        else:
            trained_collection = FaissIndexer.copy_collection(self.trained_collection)
            trained_collection_id = trained_collection["id"]
            collections = {k: FaissIndexer.copy_collection(v) for k, v in self.collections.items()}
            # deafult collection should be part of collections
            default_collection = FaissIndexer.copy_collection(self.default_collection)
            default_collection_id = default_collection["id"]
            default_collection["collection_id"] = default_collection_id
            collections[default_collection_id] = default_collection

        # only update collections
        collections_to_update = set()
        # rebuild collections
        feature_cache = {}

        for i, entry in enumerate(index_entries):
            id = entry["id"]
            collection_id = get_element(entry, "collection.id")
            if collection_id is None:
                collection_id = default_collection_id

            # mark collection
            collections_to_update.add(collection_id)

            is_public = get_element(entry, "collection.is_public")
            if is_public is None:
                is_public = False

            if is_public:
                collections_to_update.add(default_collection_id)

            for feature in entry["feature"]:

                index_name = collection_id + "." + feature["plugin"] + "." + feature["type"]
                if index_name not in feature_cache:
                    feature_cache[index_name] = []
                feature_cache[index_name].append({"id": entry["id"], "value": feature["value"]})

                if is_public:

                    index_name = default_collection_id + "." + feature["plugin"] + "." + feature["type"]
                    # print(dir(indexer_data[index_name]))
                    if index_name not in feature_cache:
                        feature_cache[index_name] = []
                    feature_cache[index_name].append({"id": entry["id"], "value": feature["value"]})

            # add features to index
            cache_to_clear = []
            for k, v in feature_cache.items():
                if len(v) >= 1000:
                    collection_id, plugin, type = k.split(".")
                    if collection_id not in collections:
                        collections[collection_id] = {
                            **FaissIndexer.copy_collection(trained_collection),
                            "collection_id": collection_id,
                            "timestamp": datetime.timestamp(datetime.now()),
                        }

                    for index in collections[collection_id]["indexes"]:
                        if index["type"] == type and index["plugin"] == plugin:
                            ids = [x["id"] for x in v]
                            values = np.asarray([x["value"] for x in v]).astype("float32")
                            faiss.normalize_L2(values)
                            index["index"].add(values)
                            index["entries"].extend(ids)
                            logging.info(f"Index: {collection_id} {index['type']} {len(index['entries'])}")
                    cache_to_clear.append(k)

            for cache in cache_to_clear:
                del feature_cache[cache]
            if i % 100 == 0:
                for _, c in collections.items():
                    logging.info(f"{c['id']} {[len(x['entries']) for x in c['indexes']]}")
        # empty the last samples
        for k, v in feature_cache.items():

            collection_id, plugin, type = k.split(".")
            if collection_id not in collections:
                collections[collection_id] = {
                    **FaissIndexer.copy_collection(trained_collection),
                    "collection_id": collection_id,
                    "timestamp": datetime.timestamp(datetime.now()),
                }

            for index in collections[collection_id]["indexes"]:
                if index["type"] == type and index["plugin"] == plugin:
                    ids = [x["id"] for x in v]
                    values = np.asarray([x["value"] for x in v]).astype("float32")
                    faiss.normalize_L2(values)
                    index["index"].add(values)
                    index["entries"].extend(ids)
                    logging.info(f"Index: {collection_id} {index['type']} {len(index['entries'])}")
            # save everything that gets updated
        new_collections = {}

        for k, collection in collections.items():
            if collection["collection_id"] in collections_to_update:
                renamed_collection = FaissIndexer.copy_collection(collection, new_ids=True)
                self.save_collection(renamed_collection)
                if collection["id"] == default_collection_id:
                    default_collection_id = renamed_collection["id"]
            new_collections[k] = renamed_collection
        collections = new_collections

        if new_trained_collection:
            self.save_collection(trained_collection)

        with open(os.path.join(self.indexer_dir, uuid.uuid4().hex + ".msg"), "wb") as f:
            f.write(
                msgpack.packb(
                    {
                        "collections": [c["id"] for _, c in collections.items()],
                        "timestamp": datetime.timestamp(datetime.now()),
                        "default_collection": default_collection_id,
                        "trained_collection": trained_collection["id"],
                    }
                )
            )
        logging.info(
            {
                "collections": [c["id"] for _, c in collections.items()],
                "timestamp": datetime.timestamp(datetime.now()),
                "default_collection": default_collection_id,
                "trained_collection": trained_collection["id"],
            }
        )

    def search(self, queries, size=100):
        result = []
        for q in queries:

            index_name = q["plugin"] + "." + q["type"]
            if index_name not in self.indexer_data:
                continue

            # TODO load it ones
            index_data = self.indexer_data[index_name]
            index = index_data["index"]
            feature = np.asarray([q["value"]]).astype("float32")
            faiss.normalize_L2(feature)
            q_result = index.search(feature, k=size)

            result.extend([index_data["rev_entries"][np.asscalar(x)] for x in q_result[1][0] if x >= 0])

        return result
