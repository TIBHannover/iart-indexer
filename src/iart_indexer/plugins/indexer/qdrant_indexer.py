import os
import re
import math
import uuid
import logging
import msgpack

from qdrant_client import QdrantClient
from qdrant_client.http import models

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


@IndexerPluginManager.export("QDrantIndexer")
class QDrantIndexer(IndexerPlugin):
    default_config = {
        "index_type": "cos",
        "host": "localhost",
        "port": 9200,
        "grpc": {"port": 50151, "host": "localhost"}
        # "indexing_size": 105536,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(QDrantIndexer, self).__init__(**kwargs)

        self.port = get_element(self.config, "port")
        if self.port is None:
            self.port = 50151

        self.host = get_element(self.config, "host")
        if self.host is None:
            self.host = "localhost"
        # Read configs
        logging.info(f"[QDrantIndexer] Connection opened")
        self.client = QdrantClient(host="localhost", port=6333, timeout=120, grpc_port=6334, prefer_grpc=False)

    def indexing(self, index_entries=None, collections=None, rebuild=False):
        logging.info(f"[QDrantIndexer] Indexing collections={collections}")
        existing_collections = [x.name for x in self.client.get_collections().collections]
        logging.info(f"[QDrantIndexer] Existing collections={existing_collections}")

        alias_uuid = uuid.uuid4().hex
        collection_batch = {}
        collection_alias_map = {}

        for i, entry in enumerate(index_entries):
            entry_id = entry["id"]
            collection_id = entry["collection"]["id"]

            collection_alias_map[collection_id] = collection_id + "_" + alias_uuid
            # Create collection if it not exist yet
            if collection_alias_map[collection_id] not in existing_collections:
                collection_dict = {}
                for feature in entry["feature"]:
                    collection_dict[feature["plugin"] + "." + feature["type"]] = models.VectorParams(
                        size=len(feature["value"]), distance=models.Distance.COSINE
                    )

                result = self.client.recreate_collection(
                    collection_name=collection_alias_map[collection_id],
                    vectors_config=collection_dict,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        ),
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=1000000000,
                    ),
                )
                existing_collections.append(collection_alias_map[collection_id])
                logging.info(
                    f"[QDrantIndexer] Create new collection {collection_id} -> {collection_alias_map[collection_id]}"
                )

            # start creating point batches for each collection
            if collection_id not in collection_batch:
                collection_batch[collection_id] = []

            point_dict = {}
            for feature in entry["feature"]:
                point_dict[feature["plugin"] + "." + feature["type"]] = (
                    feature["value"] / np.linalg.norm(feature["value"], 2)
                ).tolist()

            collection_batch[collection_id].append(
                models.PointStruct(
                    id=entry_id,
                    vector=point_dict,
                )
            )

            # check if batch size is full to flush the cache
            for collection_id, points in collection_batch.items():
                if len(points) >= 100:
                    try:
                        self.client.upsert(collection_name=collection_alias_map[collection_id], points=points)
                        collection_batch[collection_id] = []
                    except Exception as e:
                        self.client = QdrantClient(
                            host="localhost", port=6333, timeout=120, grpc_port=6334, prefer_grpc=False
                        )
                        logging.error(f"[QDrantIndexer] Insert points exception {repr(e)}")
                        continue

            if i % 1000 == 0:
                logging.info(f"[QDrantIndexer] Indexing {i} documents")

        # write the last batch
        for collection_id, points in collection_batch.items():
            if len(points) > 0:
                try:
                    self.client.upsert(collection_name=collection_alias_map[collection_id], points=points)
                except Exception as e:
                    self.client = QdrantClient(
                        host="localhost", port=6333, timeout=120, grpc_port=6334, prefer_grpc=False
                    )
                    logging.error(f"[QDrantIndexer] Insert points exception {repr(e)}")
                    continue

            self.client.update_collection(
                collection_name=collection_alias_map[collection_id],
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )

        # create an alias
        for collection_id, collection_alias_id in collection_alias_map.items():
            logging.info(
                f"[QDrantIndexer] Create new alias for collection {collection_id} -> {collection_alias_map[collection_id]}"
            )
            try:
                self.client.update_collection_aliases(
                    change_aliases_operations=[
                        models.CreateAliasOperation(
                            create_alias=models.CreateAlias(
                                collection_name=collection_alias_id, alias_name=collection_id
                            )
                        )
                    ]
                )
            except Exception as e:
                logging.error(f"[QDrantIndexer] Insert points exception {repr(e)}")
                continue

    def search(self, queries, collections=None, include_default_collection=True, size=100):
        # print(f"############################# {collections} {include_default_collection}", flush=True)
        existing_collections = [x.name for x in self.client.get_collections().collections]
        results = []
        for collection_id in collections:
            if collection_id + "_342ffe349e8f4addb0c2b49ffe467f27" not in existing_collections:
                logging.error(f"[QDrantIndexer] Unknown collection {collection_id}")
                continue
            for q in queries:
                # print(collection_id, flush=True)
                feature_name = q["plugin"] + "." + q["type"]
                feature_value = (q["value"] / np.linalg.norm(q["value"], 2)).tolist()
                result = self.client.search(
                    collection_name=collection_id + "_342ffe349e8f4addb0c2b49ffe467f27",
                    query_vector=(feature_name, feature_value),
                    limit=size,
                    search_params=models.SearchParams(hnsw_ef=512, exact=True),
                )
                results.extend(result)
                # print(f"++++++++++++++++ {result}", flush=True)

        results = [x.id for x in results]

        return results
