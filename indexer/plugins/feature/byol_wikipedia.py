from indexer.plugins import FeaturePlugin
from indexer.plugins import FeaturePluginManager
from indexer.plugins import PluginResult
import numpy as np
import math
import redisai as rai


@FeaturePluginManager.export("ByolEmbeddingFeature")
class ByolEmbeddingFeature(FeaturePlugin):
    def __init__(self, **kwargs):
        super(ByolEmbeddingFeature, self).__init__(**kwargs)
        self.host = "localhost"
        self.port = 6379
        self.model_name = "yuv_histogram"

    def call(self, entries):

        con = rai.Client(host=self.host, port=self.port)
        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            image = entry["image"]

            con.tensorset("image", image)
            result = con.modelrun(self.model_name, "image", "output")
            output = con.tensorget("output")
            uv_histogram_norm_bin = "".join([str(int(x > 0)) for x in (output / np.mean(output)).tolist()])

            hash_splits_dict = {}
            for x in range(math.ceil(len(uv_histogram_norm_bin) / 16)):
                # print(uv_histogram_norm_bin[x * 16:(x + 1) * 16])
                hash_splits_dict[f"split_{x}"] = uv_histogram_norm_bin[x * 16 : (x + 1) * 16]

            # TODO split yuv and lab color.rgb2lab
            entry_annotation.append({"type": "color", "hash": hash_splits_dict, "value": output.tolist()})

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
