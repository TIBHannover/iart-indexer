import math
import uuid
import random
import logging

import numpy as np

from sklearn.cluster import KMeans

from iart_indexer import indexer_pb2
from iart_indexer.plugins import MappingPlugin, MappingPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize


@MappingPluginManager.export("KMeansMapping")
class KMeansMapping(MappingPlugin):
    default_config = {"random_state": 42, "k": 5}

    default_version = 0.1

    def __init__(self, **kwargs):
        super(KMeansMapping, self).__init__(**kwargs)
        self.random_state = self.config["random_state"]
        self.default_k = self.config.get("k", 5)

        if self.random_state is None:
            self.random_state = random.randint()

    def call(self, entries, query, config: dict = None):
        k = self.default_k
        if isinstance(config, dict):
            try:
                k = int(config.get("k"))
            except:
                pass

        k = max(2, min(k, len(entries) - 1))
        logging.info(f"[KMeansMapping] Starting k={k}")

        kmeans = KMeans(n_clusters=k, random_state=self.random_state)
        ref_feature = "clip_embedding_feature"
        features = []
        entries = list(entries)
        for e in entries:
            for e_f in e["feature"]:
                if ref_feature != e_f["plugin"]:
                    continue
                if "value" in e_f["annotations"][0]:
                    a = e_f["annotations"][0]["value"]
                features.append(a)

        features = np.asarray(features)

        clustering = kmeans.fit_transform(features)

        logging.info(clustering)
        logging.info(kmeans.labels_)
        new_entries = [
            {**e, "cluster": kmeans.labels_[i].item(), "distance": clustering[i, kmeans.labels_[i].item()].item()}
            for i, e in enumerate(entries)
        ]
        logging.info("TEST")

        logging.info([{"c": x["cluster"], "d": x["distance"]} for x in new_entries])
        # sort in clusters and then in distance from the cluster center
        new_entries = sorted(new_entries, key=lambda x: x["distance"])
        new_entries = sorted(new_entries, key=lambda x: x["cluster"])

        logging.info("################################")
        logging.info([{"c": x["cluster"], "d": x["distance"]} for x in new_entries])
        return new_entries
