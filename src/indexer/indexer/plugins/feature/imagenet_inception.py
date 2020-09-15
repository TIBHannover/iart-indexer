from indexer.indexer.plugins import FeaturePlugin
from indexer.indexer.plugins import FeaturePluginManager
from indexer.indexer.plugins import PluginResult
import tensorflow as tf
import numpy as np
import math
import pickle
import logging


@FeaturePluginManager.export("ImageNetInceptionFeature")
class ImageNetInceptionFeature(FeaturePlugin):
    def __init__(self, **kwargs):
        super(ImageNetInceptionFeature, self).__init__(**kwargs)
        config = kwargs.get("config", {})

        if "model_path" not in config:
            logging.error("model_path not defined in config")
            return

        with tf.gfile.FastGFile(config["model_path"], "rb") as file:
            self._graph = tf.GraphDef()
            self._graph.ParseFromString(file.read())
            _ = tf.import_graph_def(self._graph, name="")

        if "pca_path" not in config:
            logging.error("pca_path not defined in config")
            return

        with open(config["pca_path"], "rb") as f:
            self._pca_params = pickle.load(f)

    def call(self, entries):
        result_entries = []
        result_annotations = []

        with tf.Session() as sess:
            for entry in entries:
                entry_annotation = []

                conv_output_tensor = sess.graph.get_tensor_by_name("pool_3:0")
                data = tf.gfile.FastGFile(entry["path"], "rb").read()
                try:
                    feature = sess.run(conv_output_tensor, {"DecodeJpeg/contents:0": data})
                    feature = np.squeeze(feature)
                except:
                    continue

                # compute pca
                feature = feature - self._pca_params.mean_
                reduc_dim_feature = np.dot(feature, self._pca_params.components_.T)
                reduc_dim_feature /= np.sqrt(self._pca_params.explained_variance_)

                # print(reduc_dim_feature.shape)
                # print(reduc_dim_feature)

                reduc_dim_feature_bin = "".join([str(int(x > 0)) for x in reduc_dim_feature.tolist()])
                hash_splits_dict = {}
                for x in range(math.ceil(len(reduc_dim_feature_bin) / 16)):
                    # print(uv_histogram_norm_bin[x * 16:(x + 1) * 16])
                    hash_splits_dict[f"split_{x}"] = reduc_dim_feature_bin[x * 16 : (x + 1) * 16]

                # TODO split yuv and lab color.rgb2lab
                entry_annotation.append(
                    {"type": "object", "hash": hash_splits_dict, "value": reduc_dim_feature.tolist()}
                )

                result_annotations.append(entry_annotation)
                result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
