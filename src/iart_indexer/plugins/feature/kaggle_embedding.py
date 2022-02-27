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
from iart_indexer.plugins import FeaturePlugin, FeaturePluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize, image_crop


@FeaturePluginManager.export("KaggleResnetFeature")
class KaggleResnetFeature(FeaturePlugin):
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
        self.host = self.config["host"]
        self.port = self.config["port"]

        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]

        self.mapping_file = self.config["mapping_file"]

        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]
        self.max_tries = self.config["max_tries"]

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

        try_count = self.max_tries
        while try_count > 0:
            try:
                self.con = rai.Client(host=self.host, port=self.port)

                if not self.check_rai():
                    self.register_rai()
                return
            except:
                try_count -= 1
                time.sleep(4)

    def register_rai(self):
        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name,
            backend="torch",
            device=self.model_device,
            data=model,
            batch=16,
        )

    def check_rai(self):
        result = self.con.modelscan()

        if self.model_name not in [x[0] for x in result]:
            return False

        return True

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

            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])

            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        type="style_embedding", binary=output_bin_str, feature=output.tolist()
                    ),
                )
            )

            output = np.squeeze(embedding_head2)

            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])

            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        type="genre_embedding", binary=output_bin_str, feature=output.tolist()
                    ),
                )
            )

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"embedding_head1_{job_id}")
            self.con.delete(f"embedding_head1_{job_id}")

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
