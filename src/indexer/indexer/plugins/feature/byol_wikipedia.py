from indexer.indexer.plugins import FeaturePlugin
from indexer.indexer.plugins import FeaturePluginManager
from indexer.indexer.plugins import PluginResult
import numpy as np
import math
import redisai as rai
import ml2rt

from indexer.indexer.utils import image_from_proto, image_resize
from indexer import indexer_pb2


@FeaturePluginManager.export("ByolEmbeddingFeature")
class ByolEmbeddingFeature(FeaturePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "byol_wikipedia",
        "model_file": "/home/matthias/byol_wikipedia.pt",
        "multicrop": True,
        "min_dim": 244,
        "max_dim": None,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(ByolEmbeddingFeature, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_file = self.config["model_file"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]

    def register_rai(self):
        con = rai.Client(host=self.host, port=self.port)
        model = ml2rt.load_model(self.model_file)

        con.modelset(
            self.model_name, backend="torch", device="cpu", data=model,
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

            con.tensorset("image", image)
            result = con.modelrun(self.model_name, "image", "output")
            output = con.tensorget("output")[0, ...]
            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])

            hash_splits_list = []
            for x in range(math.ceil(len(output_bin_str) / 16)):
                # print(uv_histogram_norm_bin[x * 16:(x + 1) * 16])
                hash_splits_list.append(output_bin_str[x * 16 : (x + 1) * 16])

            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        binary=[x.encode() for x in hash_splits_list], feature=output.tolist()
                    ),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
