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

from iart_indexer.plugins import ClassifierPlugin, ClassifierPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize
from iart_indexer import indexer_pb2


@ClassifierPluginManager.export("ImageNetResnetClassifier")
class ImageNetResnetClassifier(ClassifierPlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "imagenet_resnet",
        "model_device": "gpu",
        "model_file": "/nfs/data/iart/models/web/imagenet_resnet/imagenet_resnet.pt",
        "mapping_file": "/nfs/data/iart/models/web/imagenet_resnet/imagenet_mapping.json",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
        "threshold": 0.3,
    }

    default_version = "0.1"

    def __init__(self, **kwargs):
        super(ImageNetResnetClassifier, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.mapping_file = self.config["mapping_file"]
        self.multicrop = self.config["multicrop"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]
        self.threshold = self.config["threshold"]

        with open(self.mapping_file, "r") as f:
            self.concept_lookup = json.load(f)

        self.con = rai.Client(host=self.host, port=self.port)

        if not self.check_rai():
            self.register_rai()

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

    def call(self, entries):

        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []
            # image = image_from_proto(entry)
            image = entry
            image = image_resize(image, max_dim=self.max_dim, min_dim=self.min_dim)
            # image = np.expand_dims(image, 0)

            job_id = uuid.uuid4().hex

            self.con.tensorset(f"image_{job_id}", image)
            result = self.con.modelrun(self.model_name, f"image_{job_id}", f"probabilities_{job_id}")

            probabilities = self.con.tensorget(f"probabilities_{job_id}")

            concepts = []

            result_list = np.argwhere(probabilities[0] > self.threshold)
            for x in result_list:
                index = x[0]
                prob = probabilities[0, index]
                name = self.concept_lookup[index]
                concepts.append(indexer_pb2.Concept(concept=name, type="concept", prob=prob))

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"probabilities_{job_id}")

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
