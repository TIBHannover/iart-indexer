import math
import uuid

import time

import ml2rt
import numpy as np
import redisai as rai

from iart_indexer import indexer_pb2
from iart_indexer.plugins import ComputePlugin, ComputePluginManager, ComputePluginResult
from iart_indexer.utils import image_from_proto, image_resize, image_crop

# Preprocessing from CLIP github
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

import logging

from typing import Union, List


# @ComputePluginManager.export("ClipEmbeddingFeature")
class ClipEmbeddingFeature(ComputePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "clip_text",
        "model_device": "gpu",
        "model_file": "/home/matthias/clip_text.pt",
        "bpe_file": "/home/matthias/bpe_simple_vocab_16e6.txt.gz",
        "max_tries": 5,
    }

    default_version = "0.4"

    def __init__(self, **kwargs):
        super(ClipEmbeddingFeature, self).__init__(**kwargs)

        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]

    def check_rai(self):
        result = self.con.modelscan()
        if self.model_name in [x[0] for x in result]:
            return True
        return False

    def call(self, entries):
        result_entries = []
        result_annotations = []
        for text in entries:
            entry_annotation = []
            # image = image_from_proto(entry)
            text_imput = tokenize(self.tokenizer, text)
            job_id = uuid.uuid4().hex
            self.con.tensorset(f"text_{job_id}", text_imput)
            result = self.con.modelrun(self.model_name, f"text_{job_id}", f"output_{job_id}")
            output = self.con.tensorget(f"output_{job_id}")[0, ...]

            # normalize
            output = output / np.linalg.norm(np.asarray(output))

            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])
            self.con.delete(f"text_{job_id}")
            self.con.delete(f"output_{job_id}")

            entry_annotation.append(
                indexer_pb2.ComputePluginResult(
                    plugin=self.name,
                    type=self._type,
                    version=str(self._version),
                    feature=indexer_pb2.FeatureResult(
                        type="clip_embedding", binary=output_bin_str, feature=output.tolist()
                    ),
                )
            )

            result_annotations.append(entry_annotation)
            result_entries.append(text)

        return ComputePluginResult(self, result_entries, result_annotations)
