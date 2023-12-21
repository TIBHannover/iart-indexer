import uuid

import time

import ml2rt
import numpy as np
import redisai as rai

from iart_indexer import indexer_pb2
from iart_indexer.plugins import ComputePlugin, ComputePluginManager, ComputePluginResult

# Preprocessing from CLIP github
import os
from functools import lru_cache

import regex as re

import logging

import transformers

from typing import Union, List


# @ComputePluginManager.export("MultilingualClipEmbeddingFeature")
class MultilingualClipEmbeddingFeature(ComputePlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "multilingual_clip_text",
        "model_device": "gpu",
        "model_file": "/home/matthias/multilingual_clip_vit_b_32.pt",
        "tokenizer_model_name": "M-CLIP/XLM-Roberta-Large-Vit-B-32",
        "max_tries": 5,
    }

    default_version = "0.1"

    def __init__(self, **kwargs):
        super(MultilingualClipEmbeddingFeature, self).__init__(**kwargs)

        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.tokenizer_model_name = self.config["tokenizer_model_name"]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_model_name)

        self.max_tries = self.config["max_tries"]

        try_count = self.max_tries
        while try_count > 0:
            try:
                self.con = rai.Client(host=self.host, port=self.port)

                if not self.check_rai():
                    self.register_rai()
                return
            except:
                try_count -= 1
                time.sleep(4)

    def register_rai(self):
        model = ml2rt.load_model(self.model_file)

        print("#################", flush=True)
        print(self.model_file, flush=True)
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
        print("run")
        print(entries, flush=True)
        result_entries = []
        result_annotations = []
        for text in entries:
            entry_annotation = []
            # image = image_from_proto(entry)

            txt_tok = self.tokenizer([text], padding=True, return_tensors="np")
            # text_imput = tokenize(self.tokenizer, text)
            job_id = uuid.uuid4().hex
            self.con.tensorset(f"input_ids_{job_id}", txt_tok["input_ids"])
            self.con.tensorset(f"attention_mask_{job_id}", txt_tok["attention_mask"])
            result = self.con.modelrun(
                self.model_name, [f"input_ids_{job_id}", f"attention_mask_{job_id}"], f"output_{job_id}"
            )
            output = self.con.tensorget(f"output_{job_id}")[0, ...]

            # normalize
            print(output.shape, flush=True)
            print(output[:20], flush=True)
            print(np.mean(output), flush=True)
            print(np.var(output), flush=True)
            output = output / np.linalg.norm(np.asarray(output))
            print(np.mean(output), flush=True)
            print(np.var(output), flush=True)

            output_bin = (output > 0).astype(np.int32).tolist()
            output_bin_str = "".join([str(x) for x in output_bin])

            self.con.delete(f"input_ids_{job_id}")
            self.con.delete(f"attention_mask_{job_id}")
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
