import os
import math
import uuid
import pickle
import logging

try:
    import faiss
except ImportError as error:
    pass

import ml2rt
import numpy as np
import redisai as rai

from iart_indexer import indexer_pb2
from iart_indexer.plugins import IndexerPlugin, IndexerPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize


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

        os.makedirs(self.indexer_dir, exist_ok=True)

        self.load()

    def load(self):
        # load data
        self.indexer_data = {}
        self.trained_index = None
        self.collection_index_map = {}
        self.default_collections = []

        if self.indexer_dir is not None and os.path.isfile(os.path.join(self.indexer_dir, "trained_index.index")):
            self.trained_index = faiss.read_index(os.path.join(self.indexer_dir, "trained_index.index"))

        if self.indexer_dir is not None and os.path.isfile(os.path.join(self.indexer_dir, "default_collections.index")):
            self.default_collections = faiss.read_index(os.path.join(self.indexer_dir, "default_collections.index"))

        if self.indexer_dir is not None and os.path.isfile(os.path.join(self.indexer_dir, "collections.pkl")):
            logging.info(f"FaissIndexer: Found collection map")
            with open(os.path.join(self.indexer_dir, "collections.pkl"), "rb") as f:
                self.collection_index_map = pickle.load(f)

            for k, v in self.collection_index_map.items():

                if self.indexer_dir is not None and os.path.isfile(os.path.join(self.indexer_dir, f"{v}.pkl")):
                    logging.info(f"FaissIndexer: Found collection map")
                    with open(os.path.join(self.indexer_dir, f"{v}.pkl"), "rb") as f:
                        self.indexer_data[v] = pickle.load(f)

                for key, index_data in self.indexer_data[v].items():
                    logging.info(f"FaissIndexer: Restore index {key}")
                    self.indexer_data[v][key]["index"] = faiss.read_index(
                        os.path.join(self.indexer_dir, index_data["id"] + ".index")
                    )
                    self.indexer_data[v][key]["index"].nprobe = self.number_of_probs

    def train(self, entries):
        data = {}
        index_names = set()
        for i, entry in enumerate(entries):
            id = entry["id"]

            for feature in entry["feature"]:
                index_name = feature["plugin"] + "." + feature["type"]

                if index_name not in data:
                    data[index_name] = {"entries": {}, "data": [], "d": 0, "id": uuid.uuid4().hex}
                data[index_name]["data"].append(feature["value"])
                data[index_name]["d"] = len(feature["value"])

                if id not in data[index_name]["entries"]:
                    data[index_name]["entries"][id] = len(data[index_name]["entries"])

                index_names.add(index_name)

            if i % 1000 == 0 and i > 0:
                logging.info(f"[FaissIndexer] Read {i}")

                for index_name, index_data in data.items():
                    logging.info(f"[FaissIndexer] {index_name}:{len(data[index_name]['data'])}")
            if i > self.train_size:
                break
                # break

        logging.info(f"[FaissIndexer] Start training")
        indexer_data = {}
        for index_name, index_data in data.items():
            d = index_data["d"]
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.number_of_cluster)

            train_data = np.asarray(index_data["data"]).astype("float32")
            faiss.normalize_L2(train_data)
            index.train(train_data)
            # index.add(train_data)

            # faiss.write_index(index, os.path.join(self.indexer_dir, index_data["id"] + ".index"))
            indexer_data[index_name] = {
                "index": index,
                "entries": {},
                "data": [],
                "id": uuid.uuid4().hex,
                # "rev_entries": {v: k for k, v in index_data["entries"].items()},
                "d": index_data["d"],
                # "id": index_data["id"],
            }
        return indexer_data

    def indexing(self, train_entries, index_entries, rebuild=False):
        logging.info(f"Faiss: indexing (rebuild={rebuild})")
        if rebuild:
            trained_index = self.train(train_entries)
            collection_index_map = {}
            indexer_data = {}
            default_collections = []
        else:
            trained_index = self.trained_index
            collection_index_map = self.collection_index_map
            indexer_data = self.indexer_data
            default_collections = self.default_collections

        logging.info(indexer_data.keys())
        logging.info(collection_index_map)
        # indexer_data = {}

        for i, entry in enumerate(index_entries):
            id = entry["id"]
            collection = (
                "default" if entry.get("collection", False) else entry.get("collection", {}).get("id", "default")
            )
            default = True if entry.get("collection", False) else entry.get("collection", {}).get("is_public", False)
            logging.info(entry.keys())
            logging.info(entry.get("collection"))
            logging.info(id)
            logging.info(collection)
            logging.info(default)
            for feature in entry["feature"]:

                index_name = feature["plugin"] + "." + feature["type"]
                # print(dir(indexer_data[index_name]))

                indexer_data[index_name]["data"].append(feature["value"])

                if id not in indexer_data[index_name]["entries"]:
                    indexer_data[index_name]["entries"][id] = len(indexer_data[index_name]["entries"])

            if i % 1000 == 0 and i > 0:
                logging.info(f"[FaissIndexer] Read {i} ")

                for index_name, index_data in indexer_data.items():
                    logging.info(f"[FaissIndexer] {index_name}:{len(indexer_data[index_name]['data'])}")
                    if len(indexer_data[index_name]["data"]) == 0:
                        continue
                    train_data = np.asarray(index_data["data"]).astype("float32")
                    # index_data["index"].train(train_data)
                    faiss.normalize_L2(train_data)
                    index_data["index"].add(train_data)

                    indexer_data[index_name]["data"] = []

        output_indexer_data = {}
        for index_name, index_data in indexer_data.items():
            d = index_data["d"]

            faiss.write_index(index_data["index"], os.path.join(self.indexer_dir, index_data["id"] + ".index"))
            indexer_data[index_name] = {
                "entries": index_data["entries"],
                "rev_entries": {v: k for k, v in index_data["entries"].items()},
                "d": index_data["d"],
                "id": index_data["id"],
            }

        with open(os.path.join(self.indexer_dir, "data.pkl"), "wb") as f:
            pickle.dump(indexer_data, f)

        # TODO check multithreading
        if self.indexer_dir is not None and os.path.isfile(os.path.join(self.indexer_dir, "data.pkl")):
            with open(os.path.join(self.indexer_dir, "data.pkl"), "rb") as f:
                self.indexer_data = pickle.load(f)

        for key, index_data in self.indexer_data.items():
            self.indexer_data[key]["index"] = faiss.read_index(
                os.path.join(self.indexer_dir, index_data["id"] + ".index")
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
