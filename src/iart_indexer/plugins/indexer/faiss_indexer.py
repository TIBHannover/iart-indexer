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
        "number_of_cluster": 100
        # "indexing_size": 105536,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(FaissIndexer, self).__init__(**kwargs)

        self.indexer_dir = self.config["indexer_dir"]
        self.train_size = self.config["train_size"]
        self.number_of_cluster = self.config["number_of_cluster"]
        # self.indexing_size = self.config["indexing_size"]

        self.indexer_data = {}
        if self.indexer_dir is not None and os.path.isfile(os.path.join(self.indexer_dir, "data.pkl")):
            logging.info(f"FaissIndexer: Found index")
            with open(os.path.join(self.indexer_dir, "data.pkl"), "rb") as f:
                self.indexer_data = pickle.load(f)

            for key, index_data in self.indexer_data.items():
                logging.info(f"FaissIndexer: Restore index {key}")
                self.indexer_data[key]["index"] = faiss.read_index(
                    os.path.join(self.indexer_dir, index_data["id"] + ".index")
                )

        os.makedirs(self.indexer_dir, exist_ok=True)

    def train(self, entries):
        data = {}
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

            if i % 10000 == 0 and i > 0:
                logging.info(f"FaissIndexer: Read {i}")
            if i > self.train_size:
                break
                # break

        logging.info(f"FaissIndexer: Start training")
        indexer_data = {}
        for index_name, index_data in data.items():
            d = index_data["d"]
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.number_of_cluster)

            train_data = np.asarray(index_data["data"]).astype("float32")
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

    def indexing(self, entries):
        indexer_data = self.train(entries)
        # indexer_data = {}
        for i, entry in enumerate(entries):
            id = entry["id"]

            for feature in entry["feature"]:
                index_name = feature["plugin"] + "." + feature["type"]
                # print(dir(indexer_data[index_name]))
                indexer_data[index_name]["data"].append(feature["value"])

                if id not in indexer_data[index_name]["entries"]:
                    indexer_data[index_name]["entries"][id] = len(indexer_data[index_name]["entries"])

            if i % 10000 == 0 and i > 0:
                logging.info(f"FaissIndexer: Read {i} ")

                for index_name, index_data in indexer_data.items():
                    logging.info(f"{len(indexer_data[index_name]['data'])}")
                    train_data = np.asarray(index_data["data"]).astype("float32")
                    # index_data["index"].train(train_data)
                    index_data["index"].add(train_data)

                    indexer_data[index_name]["data"] = []
            # if i > self.indexing_size:
            #     break

            # break

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
            q_result = index.search(np.asarray([q["value"]]).astype("float32"), k=size)

            result.extend([index_data["rev_entries"][np.asscalar(x)] for x in q_result[1][0] if x >= 0])

        return result
