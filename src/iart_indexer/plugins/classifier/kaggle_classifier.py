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

from iart_indexer.plugins import ClassifierPlugin, ClassifierPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize
from iart_indexer import indexer_pb2


@ClassifierPluginManager.export("KaggleResnetClassifier")
class KaggleResnetClassifier(ClassifierPlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "kaggle_resnet_classifier",
        "model_device": "gpu",
        "model_file": "/nfs/data/iart/models/web/kaggle_classifier/kaggle_classifier_gpu.pt",
        "mapping_file": "/nfs/data/iart/models/web/kaggle_classifier/mapping.json",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
        "threshold": 0.25,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(KaggleResnetClassifier, self).__init__(**kwargs)
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

            result = self.con.modelrun(
                self.model_name, f"image_{job_id}", [f"probabilities_head1_{job_id}", f"probabilities_head2_{job_id}"]
            )

            probabilities_head1 = self.con.tensorget(f"probabilities_head1_{job_id}")
            probabilities_head2 = self.con.tensorget(f"probabilities_head2_{job_id}")

            concepts = []

            result_list = np.argwhere(probabilities_head1[0] > self.threshold)
            for x in result_list:
                index = x[0]
                prob = probabilities_head1[0, index]
                name = self.mappings[0]["mappings"][index]["name"]
                concepts.append(indexer_pb2.Concept(concept=name, type="style", prob=prob))

            result_list = np.argwhere(probabilities_head2[0] > self.threshold)
            for x in result_list:
                index = x[0]
                prob = probabilities_head2[0, index]
                name = self.mappings[1]["mappings"][index]["name"]
                concepts.append(indexer_pb2.Concept(concept=name, type="genre", prob=prob))

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"probabilities_head1_{job_id}")
            self.con.delete(f"probabilities_head2_{job_id}")

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
