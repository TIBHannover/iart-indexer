import math
import uuid
import time

import ml2rt
import numpy as np
import redisai as rai

import pickle
import json
import logging

from iart_indexer import indexer_pb2
from iart_indexer.plugins import ComputePlugin, ComputePluginManager, ComputePluginResult
from iart_indexer.utils import image_from_proto, image_resize, image_crop


# @ComputePluginManager.export("KaggleResnetFeature")
class KaggleResnetFeature(ComputePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "kaggle_resnet_embedding",
        "model_device": "gpu",
        "model_file": "/nfs/data/iart/models/web/kaggle_embedding/kaggle_embedding_gpu.pt",
        "mapping_file": "/nfs/data/iart/models/web/kaggle_embedding/mapping.json",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
        "max_tries": 5,
    }

    default_version = "1.3"

    def __init__(self, **kwargs):
        super(KaggleResnetFeature, self).__init__(**kwargs)

        self.mapping_file = self.config["mapping_file"]

        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]

        self.heads_name = ["style", "genre"]
        self.mappings = []

        for h in self.heads_name:
            self.mappings.append({"head": h, "mappings": []})

        with open(self.mapping_file, "r") as f:
            for line in f:
                d = json.loads(line)
                for m in self.mappings:
                    if m["head"] == d["type"]:
                        m["mappings"].append(d)

    def call(self, entries):
        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            # image = image_from_proto(entry)
            image = entry
            image = image_resize(image, max_dim=self.max_dim, min_dim=self.min_dim)
            # image = image_crop(image, [224, 224])

            # image = np.expand_dims(image, axis=0)  # / 256
            # image = image.astype(np.float32)

            job_id = uuid.uuid4().hex

            self.con.tensorset(f"image_{job_id}", image)
            result = self.con.modelrun(
                self.model_name, f"image_{job_id}", [f"embedding_head1_{job_id}", f"embedding_head2_{job_id}"]
            )
            embedding_head1 = self.con.tensorget(f"embedding_head1_{job_id}")[0, ...]
            embedding_head2 = self.con.tensorget(f"embedding_head2_{job_id}")[0, ...]

            output = np.squeeze(embedding_head1)

            entry_annotation.append(
                indexer_pb2.ComputePluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(type="style_embedding", feature=output.tolist()),
                )
            )

            output = np.squeeze(embedding_head2)

            entry_annotation.append(
                indexer_pb2.ComputePluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(type="genre_embedding", feature=output.tolist()),
                )
            )

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"embedding_head1_{job_id}")
            self.con.delete(f"embedding_head1_{job_id}")

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return ComputePluginResult(self, result_entries, result_annotations)
