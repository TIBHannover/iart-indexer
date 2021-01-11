import math
import uuid

import ml2rt
import numpy as np
import redisai as rai

from iart_indexer import indexer_pb2
from iart_indexer.plugins import FeaturePlugin, FeaturePluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize


@FeaturePluginManager.export("YUVHistogramFeature")
class YUVHistogramFeature(FeaturePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "yuv_histogram",
        "model_file": "/home/matthias/yuv_histogram.pt",
        "max_dim": 244,
        "min_dim": 244,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(YUVHistogramFeature, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_file = self.config["model_file"]

        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]

        self.con = rai.Client(host=self.host, port=self.port)

        if not self.check_rai():
            self.register_rai()

    def register_rai(self):
        model = ml2rt.load_model(self.model_file)

        self.con.modelset(
            self.model_name, backend="torch", device="cpu", data=model,
        )

    def check_rai(self):
        result = self.con.modelscan()
        if self.model_name in [x[0] for x in result]:
            return True
        return False

    def call(self, entries):
        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            # image = image_from_proto(entry)
            image = entry
            image = image_resize(image, max_dim=self.max_dim)
            image = image.astype(np.float32)

            job_id = uuid.uuid4().hex

            self.con.tensorset(f"image_{job_id}", image)
            result = self.con.modelrun(self.model_name, f"image_{job_id}", f"output_{job_id}")
            output = self.con.tensorget(f"output_{job_id}")
            uv_histogram_norm_bin = "".join([str(int(x > 0)) for x in (output / np.mean(output)).tolist()])

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"output_{job_id}")

            # hash_splits_list = []
            # for x in range(math.ceil(len(uv_histogram_norm_bin) / 16)):
            #     # print(uv_histogram_norm_bin[x * 16:(x + 1) * 16])
            #     hash_splits_list.append(uv_histogram_norm_bin[x * 16 : (x + 1) * 16])

            # TODO split yuv and lab color.rgb2lab
            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        type="color", binary=uv_histogram_norm_bin, feature=output.tolist()
                    ),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
