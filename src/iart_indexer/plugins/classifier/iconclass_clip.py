import logging
import os
import re
import sys
import threading
import uuid

import grpc
import json

import numpy as np
import math
import redisai as rai
import ml2rt
import csv
import pickle

import scipy

from iart_indexer.plugins import ClassifierPlugin, ClassifierPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize, image_crop
from iart_indexer import indexer_pb2


@ClassifierPluginManager.export("IconclassCLIPClassifier")
class IconclassCLIPClassifier(ClassifierPlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "iconclass_clip",
        "model_device": "gpu",
        "model_file": "/nfs/data/iart/web/models/iconclass_clip/clip_image_gpu.pt",
        "txt_embedding_file": "/nfs/data/iart/web/models/iconclass_clip/text_kw_txt_embedding.pl",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
        "threshold": 2e-2,
        "k": 25,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(IconclassCLIPClassifier, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.txt_embedding_file = self.config["txt_embedding_file"]
        self.multicrop = self.config["multicrop"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]
        self.threshold = self.config["threshold"]
        self.k = self.config["k"]

        self.concept_lookup = []

        with open(self.txt_embedding_file, "rb") as f:
            self.txt_mapping = pickle.load(f)
            self.txt_feature = np.concatenate([x["clip"] for x in self.txt_mapping])

        self.con = rai.Client(host=self.host, port=self.port)

        if not self.check_rai():
            self.register_rai()

    def get_top_k(self, img_features, txt_features):
        if isinstance(img_features, list):
            img_features = np.array(img_features)

        features = img_features @ txt_features.T

        result = scipy.special.softmax(100.0 * features, axis=1)

        top_k = np.argpartition(result, -self.k, axis=-1)[:, -self.k :]

        scores = []
        indexes = []
        for l in range(top_k.shape[0]):
            values = result[l, top_k[l, ...]]
            indices = top_k[l, ...]

            scores.append(values)
            indexes.append(indices)

        return np.stack(scores), np.stack(indexes)

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
        if self.model_name in [x[0] for x in result]:
            return True
        return False

    def generate_label(self, entry):
        seq = entry["id"]
        tags = ", ".join(entry["kw"]["en"])
        desc = entry["txt"]["en"]
        return f"{seq}; {tags}; {desc}"

    def call(self, entries):

        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            # image = image_from_proto(entry)

            image = entry
            image = image_resize(image, max_dim=self.max_dim, min_dim=self.min_dim)
            image = image_crop(image, [224, 224])

            job_id = uuid.uuid4().hex

            self.con.tensorset(f"image_{job_id}", image)
            result = self.con.modelrun(self.model_name, f"image_{job_id}", f"output_{job_id}")
            output = self.con.tensorget(f"output_{job_id}")
            self.con.delete(f"image_{job_id}")
            self.con.delete(f"output_{job_id}")

            probs, indexes = self.get_top_k(output, self.txt_feature)
            # print(lb_seqs)
            # probabilities = np.exp(probabilities)
            # print(probabilities)
            concepts = []

            # print(seqs)

            for i in range(indexes.shape[1]):
                if probs[0, i] < self.threshold:
                    continue
                index = indexes[0, i].item()
                # print(f"{i} {index} {probs.shape} {indexes.shape}", flush=True)
                meta = self.txt_mapping[index]
                # print()
                name = self.generate_label(meta)
                concepts.append(indexer_pb2.Concept(concept=name, type="concept", prob=probs[0, i].item()))

            entry_annotation.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    classifier=indexer_pb2.ClassifierResult(concepts=concepts),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)


# import math
# import uuid


# import ml2rt
# import numpy as np
# import redisai as rai

# from iart_indexer import indexer_pb2
# from iart_indexer.plugins import FeaturePlugin, FeaturePluginManager, PluginResult
# from iart_indexer.utils import image_from_proto, image_resize, image_crop


# @FeaturePluginManager.export("ClipEmbeddingFeature")
# class ClipEmbeddingFeature(FeaturePlugin):
#     default_config = {
#         "host": "localhost",
#         "port": 6379,
#         "model_name": "clip_image",
#         "model_device": "gpu",
#         "model_file": "/home/matthias/clip_image.pt",
#         "multicrop": True,
#         "max_dim": None,
#         "min_dim": 224,
#     }

#     default_version = 0.3

#     def __init__(self, **kwargs):
#         super(ClipEmbeddingFeature, self).__init__(**kwargs)
#         self.host = self.config["host"]
#         self.port = self.config["port"]
#         self.model_name = self.config["model_name"]
#         self.model_device = self.config["model_device"]
#         self.model_file = self.config["model_file"]
#         self.max_dim = self.config["max_dim"]
#         self.min_dim = self.config["min_dim"]

#         self.con = rai.Client(host=self.host, port=self.port)

#         if not self.check_rai():
#             self.register_rai()

#     def register_rai(self):
#         model = ml2rt.load_model(self.model_file)

#         self.con.modelset(
#             self.model_name,
#             backend="torch",
#             device=self.model_device,
#             data=model,
#             batch=16,
#         )

#     def check_rai(self):
#         result = self.con.modelscan()
#         if self.model_name in [x[0] for x in result]:
#             return True
#         return False

#     def call(self, entries):

#         result_entries = []
#         result_annotations = []
#         for entry in entries:
#             entry_annotation = []
#             # image = image_from_proto(entry)
#             image = entry
#             image = image_resize(image, max_dim=self.max_dim, min_dim=self.min_dim)
#             image = image_crop(image, [224, 224])

#             job_id = uuid.uuid4().hex

#             self.con.tensorset(f"image_{job_id}", image)
#             result = self.con.modelrun(self.model_name, f"image_{job_id}", f"output_{job_id}")
#             output = self.con.tensorget(f"output_{job_id}")[0, ...]
#             output_bin = (output > 0).astype(np.int32).tolist()
#             output_bin_str = "".join([str(x) for x in output_bin])

#             self.con.delete(f"image_{job_id}")
#             self.con.delete(f"output_{job_id}")

#             entry_annotation.append(
#                 indexer_pb2.PluginResult(
#                     plugin=self.name,
#                     type=self._type,
#                     version=str(self._version),
#                     feature=indexer_pb2.FeatureResult(
#                         type="clip_embedding", binary=output_bin_str, feature=output.tolist()
#                     ),
#                 )
#             )

#             result_annotations.append(entry_annotation)
#             result_entries.append(entry)

#         return PluginResult(self, result_entries, result_annotations)
