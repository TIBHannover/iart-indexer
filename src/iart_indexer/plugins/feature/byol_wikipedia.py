import math
import uuid


import ml2rt
import numpy as np
import redisai as rai

from iart_indexer import indexer_pb2
from iart_indexer.plugins import FeaturePlugin, FeaturePluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize


@FeaturePluginManager.export("ByolEmbeddingFeature")
class ByolEmbeddingFeature(FeaturePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "byol_wikipedia",
        "model_device": "gpu",
        "model_file": "/home/matthias/byol_wikipedia.pt",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(ByolEmbeddingFeature, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]

    def register_rai(self):
        con = rai.Client(host=self.host, port=self.port)
        model = ml2rt.load_model(self.model_file)

        con.modelset(
            self.model_name, backend="torch", device=self.model_device, data=model, batch=16,
        )

    def check_rai(self):
        con = rai.Client(host=self.host, port=self.port)
        result = con.modelscan()
        if self.model_name in [x[0] for x in result]:
            return True
        return False

    def call(self, entries):

        if not self.check_rai():
            self.register_rai()

        con = rai.Client(host=self.host, port=self.port)
        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            image = image_from_proto(entry)
            image = image_resize(image, max_dim=self.max_dim, min_dim=self.min_dim)

            job_id = uuid.uuid4().hex

            con.tensorset(f"image_{job_id}", image)
            result = con.modelrun(self.model_name, f"image_{job_id}", f"output_{job_id}")
            output = con.tensorget(f"output_{job_id}")[0, ...]
            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])

            con.delete(f"image_{job_id}")
            con.delete(f"output_{job_id}")

            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        type="byol_embedding", binary=output_bin_str, feature=output.tolist()
                    ),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)