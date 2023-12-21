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

from iart_indexer.plugins import ComputePlugin, ComputePluginManager, ComputePluginResult
from iart_indexer.utils import image_from_proto, image_resize
from iart_indexer import indexer_pb2


# @ComputePluginManager.export("IconclassLSTMClassifier")
class IconclassLSTMClassifier(ComputePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "iconclass_lstm",
        "model_device": "gpu",
        "model_file": "/nfs/data/iart/models/web/iconclass_lstm/kaggle_embedding_gpu.pt",
        "mapping_file": "/nfs/data/iart/web/models/iconclass_lstm/labels.jsonl",
        "classifier_file": "/nfs/data/iart/web/models/iconclass_lstm/classifiers.jsonl",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
        "threshold": 1e-3,
    }

    default_version = "0.1"

    def __init__(self, **kwargs):
        super(IconclassLSTMClassifier, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.mapping_file = self.config["mapping_file"]
        self.classifier_file = self.config["classifier_file"]
        self.multicrop = self.config["multicrop"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]
        self.threshold = self.config["threshold"]

        self.concept_lookup = []

        self.mapping_data = self.read_jsonl(self.mapping_file)

        self.mapping = {}
        for i in self.mapping_data:
            self.mapping[i["id"]] = i

        self.classifier_config = {}
        if self.classifier_file is not None:
            self.classifier_config = self.read_jsonl(self.classifier_file)

        self.con = rai.Client(host=self.host, port=self.port)

        if not self.check_rai():
            self.register_rai()

    def seq2str(self, seqs):
        # reverse the indexes in dictionary to the string labels
        final_lbs = list()
        for seq in seqs:
            label_hirarchy = list()
            for i_lev in range(len(seq)):
                if seq[i_lev] != 0:
                    label_hirarchy.append(self.classifier_config[i_lev]["tokenizer"][seq[i_lev]])

            # print(''.join(label_hirarchy))
            final_lbs.append("".join(label_hirarchy))
        return final_lbs

    def get_element(self, data_dict: dict, path: str, split_element="."):
        if path is None:
            return data_dict

        if callable(path):
            elem = path(data_dict)

        if isinstance(path, str):
            elem = data_dict
            try:
                for x in path.strip(split_element).split(split_element):
                    try:
                        x = int(x)
                        elem = elem[x]
                    except ValueError:
                        elem = elem.get(x)
            except:
                pass

        if isinstance(path, (list, set)):
            elem = [self.get_element(data_dict, x) for x in path]

        return elem

    def read_jsonl(self, path, dict_key=None, keep_keys=None):
        data = []
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)
                if keep_keys is not None:
                    d = {k: self.get_element(d, v) for k, v in keep_keys.items()}
                data.append(d)

        if dict_key is not None:
            data = {self.get_element(x, dict_key): x for x in data}

        return data

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
        seq = entry["seq"]
        tags = ", ".join(entry["meta"]["kw"]["en"])
        desc = entry["meta"]["txt"]["en"]
        return f"{seq}; {tags}; {desc}"

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
            result = self.con.modelrun(self.model_name, f"image_{job_id}", [f"seqs_{job_id}", f"probs_{job_id}"])

            seqs = self.con.tensorget(f"seqs_{job_id}")
            probabilities = self.con.tensorget(f"probs_{job_id}")

            seqs = [{"seq": self.seq2str([s[1:]])[0], "prob": np.exp(probabilities[i])} for i, s in enumerate(seqs)]

            seqs = [{**x, "meta": self.mapping[x["seq"]]} for x in seqs if x["seq"] in self.mapping]
            # final_beams = list(map(lambda l: l[1:], final_beams))
            # seqs = self.seq2str(final_beams)
            seqs = sorted(seqs, key=lambda x: len(x["seq"]))
            seqs = [j for i, j in enumerate(seqs) if all(j["seq"] not in k["seq"] for k in seqs[i + 1 :])]

            # print(lb_seqs)
            # probabilities = np.exp(probabilities)
            # print(probabilities)
            concepts = []

            # print(seqs)

            for x in seqs:
                if x["prob"] < self.threshold:
                    continue

                name = self.generate_label(x)
                concepts.append(indexer_pb2.Concept(concept=name, type="concept", prob=x["prob"].item()))

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"seqs_{job_id}")
            self.con.delete(f"probs_{job_id}")

            entry_annotation.append(
                indexer_pb2.ComputePluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    classifier=indexer_pb2.ClassifierResult(concepts=concepts),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return ComputePluginResult(self, result_entries, result_annotations)
