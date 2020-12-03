import math
import uuid

import numpy as np

import umap
import random

from sklearn.preprocessing import MinMaxScaler

from iart_indexer import indexer_pb2
from iart_indexer.plugins import MappingPlugin, MappingPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize


@MappingPluginManager.export("UMapMapping")
class UMapMapping(MappingPlugin):
    default_config = {"random_state": 42, "n_neighbors": 3, "min_dist": 0.1}

    default_version = 0.1

    def __init__(self, **kwargs):
        super(UMapMapping, self).__init__(**kwargs)

        self.random_state = self.config["random_state"]
        self.n_neighbors = self.config["n_neighbors"]
        self.min_dist = self.config["min_dist"]

        if self.random_state is None:
            self.random_state = random.randint()

        self.reducer = umap.UMAP(random_state=self.random_state, n_neighbors=self.n_neighbors, min_dist=self.min_dist)

        self.scaler = MinMaxScaler()

    def call(self, entries, query):
        ref_feature = "byol_embedding_feature"
        features = []
        entries = list(entries)
        for e in entries:
            for e_f in e["feature"]:
                if ref_feature != e_f["plugin"]:
                    continue
                if "val_64" in e_f["annotations"][0]:
                    a = e_f["annotations"][0]["val_64"]
                if "val_128" in e_f["annotations"][0]:
                    a = e_f["annotations"][0]["val_128"]
                if "val_256" in e_f["annotations"][0]:
                    a = e_f["annotations"][0]["val_256"]
                features.append(a)
        features = np.asarray(features)
        embedding = self.reducer.fit_transform(features)
        normalize_embeddings = self.scaler.fit_transform(embedding)
        new_entries = [{**e, "coordinates": normalize_embeddings[i, :].tolist()} for i, e in enumerate(entries)]

        return new_entries
