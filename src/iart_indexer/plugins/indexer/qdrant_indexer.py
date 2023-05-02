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
        self.client = QdrantClient(host="localhost", port=6333)

    def indexing(self, index_entries=None, collections=None, rebuild=False):
        logging.info(f"[QDrantIndexer] Indexing collections={collections}")
        existing_collections = [x.name for x in self.client.get_collections().collections]
        logging.info(f"[QDrantIndexer] Existing collections={existing_collections}")

        for i, entry in enumerate(index_entries):
            print(i, flush=True)
            entry_id = entry["id"]
            collection_id = entry["collection"]["id"]
            if collection_id not in existing_collections:
                collection_dict = {}
                for feature in entry["feature"]:
                    collection_dict[feature["plugin"] + "." + feature["type"]] = models.VectorParams(
                        size=len(feature["value"]), distance=models.Distance.COSINE
                    )

                print(collection_dict)
                result = self.client.recreate_collection(
                    collection_name=collection_id,
                    vectors_config=collection_dict,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        ),
                    ),
                )
                print(result)
                existing_collections.append(collection_id)

            point_dict = {}
            for feature in entry["feature"]:
                point_dict[feature["plugin"] + "." + feature["type"]] = feature["value"]
            self.client.upsert(
                collection_name=collection_id,
                points=[
                    models.PointStruct(
                        id=entry_id,
                        vector=point_dict,
                    ),
                ],
            )

    def search(self, queries, collections=None, include_default_collection=True, size=100):
        # print(f"############################# {collections} {include_default_collection}", flush=True)
        existing_collections = [x.name for x in self.client.get_collections().collections]
        results = []
        for collection_id in collections:
            if collection_id not in existing_collections:
                logging.error(f"[QDrantIndexer] Unknown collection {collection_id}")
                continue
            for q in queries:
                # print(collection_id, flush=True)
                feature_name = q["plugin"] + "." + q["type"]
                feature_value = q["value"]
                result = self.client.search(
                    collection_name=collection_id,
                    query_vector=(feature_name, feature_value),
                    limit=size,
                )
                results.extend(result)
                # print(f"++++++++++++++++ {result}", flush=True)

        results = [x.id for x in results]

        return results
