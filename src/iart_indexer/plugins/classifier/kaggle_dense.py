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


@ClassifierPluginManager.export("KaggleDenseClassifier")
class KaggleDenseClassifier(ClassifierPlugin):
    default_config = {
        "host": "localhost",
        "port": 6379,
        "model_name": "kaggle_densenet",
        "model_device": "gpu",
        "model_file": "/nfs/data/iart/models/web/kaggle_painter/kaggle_densenet.pb",
        "mapping_file": "/nfs/data/iart/models/web/kaggle_painter/kaggle_mapping.json",
        "multicrop": True,
        "max_dim": None,
        "min_dim": 244,
        "threshold": 0.5,
    }

    default_version = 0.1

    def __init__(self, **kwargs):
        super(KaggleDenseClassifier, self).__init__(**kwargs)
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.model_name = self.config["model_name"]
        self.model_device = self.config["model_device"]
        self.model_file = self.config["model_file"]
        self.mapping_file = self.config["mapping_file"]
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]
        self.threshold = self.config["threshold"]

        with open(self.mapping_file, "r") as f:
            self.mapping = json.load(f)

        self.name_to_tensor = {
            "genre_probabilities": "softmax_tensor_genre",
            # "genre_name": "hash_table_Lookup_2/LookupTableFindV2",
            "style_probabilities": "softmax_tensor_style",
            # "style_name": "hash_table_Lookup_1/LookupTableFindV2",
        }

        self.outputs = [
            "genre_probabilities",
            # "genre_name",
            "style_probabilities",
            # "style_name"
        ]

        self.genre_lookup = sorted(
            [(token, index) for token, index in self.mapping["genres"].items()], key=lambda x: x[1]
        )

        self.style_lookup = sorted(
            [(token, index) for token, index in self.mapping["styles"].items()], key=lambda x: x[1]
        )

        self.con = rai.Client(host=self.host, port=self.port)

        if not self.check_rai():
            self.register_rai()

    def register_rai(self):
        model = ml2rt.load_model(self.model_file)

        outputs = [
            "genre_probabilities",
            # "genre_name",
            "style_probabilities",
            # "style_name"
        ]

        self.con.modelset(
            self.model_name,
            backend="TF",
            device=self.model_device,
            data=model,
            inputs=["image"],
            outputs=[self.name_to_tensor[x] for x in self.outputs],
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
            image = np.expand_dims(image, 0)

            job_id = uuid.uuid4().hex

            self.con.tensorset(f"image_{job_id}", image)
            result = self.con.modelrun(
                self.model_name, f"image_{job_id}", [f"genre_probabilities_{job_id}", f"style_probabilities_{job_id}"]
            )

            genre_probabilities = self.con.tensorget(f"genre_probabilities_{job_id}")
            style_probabilities = self.con.tensorget(f"style_probabilities_{job_id}")

            concepts = []

            genre_best = np.argmax(genre_probabilities[0]).item()
            genre = self.genre_lookup[genre_best]
            genre_prob = genre_probabilities[0, genre_best].item()

            if genre_prob > self.threshold:
                concepts.append(indexer_pb2.Concept(concept=genre[0], type="genre", prob=genre_prob))

            style_best = np.argmax(style_probabilities[0]).item()
            style = self.style_lookup[style_best]
            style_prob = style_probabilities[0, style_best].item()

            if style_prob > self.threshold:
                concepts.append(indexer_pb2.Concept(concept=style[0], type="style", prob=style_prob))

            self.con.delete(f"image_{job_id}")
            self.con.delete(f"genre_probabilities_{job_id}")
            self.con.delete(f"style_probabilities_{job_id}")
            #     # decode results
            #     artist_name = np.array(results.outputs["artist_name"].string_val)
            #     artist = np.array(results.outputs["artist"].int64_val)
            #     artist_probabilities = np.array(results.outputs["artist_probabilities"].float_val)
            #     genre_name = np.array(results.outputs["genre_name"].string_val)
            #     genre = np.array(results.outputs["genre"].int64_val)
            #     genre_probabilities = np.array(results.outputs["genre_probabilities"].float_val)
            #     style_name = np.array(results.outputs["style_name"].string_val)
            #     style = np.array(results.outputs["style"].int64_val)
            #     style_probabilities = np.array(results.outputs["style_probabilities"].float_val)

            #     if genre_probabilities[genre] > self.config["threshold"]:
            #         entry_annotation.append(
            #             {
            #                 "type": "genre",
            #                 "name": genre_name.item().decode("utf-8"),
            #                 "value": genre_probabilities[genre].item(),
            #             }
            #         )

            #     if style_probabilities[style] > self.config["threshold"]:
            #         entry_annotation.append(
            #             {
            #                 "type": "style",
            #                 "name": style_name.item().decode("utf-8"),
            #                 "value": style_probabilities[style].item(),
            #             }
            #         )

            # result_annotations.append(entry_annotation)
            # result_entries.append(entry)

            # output = con.tensorget("output")[0, ...]
            # output_bin = (output > 0).astype(np.int32).tolist()
            # output_bin_str = "".join([str(x) for x in output_bin])

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


# @ClassifierPluginManager.export("KaggleDenseClassifier")
# class KaggleDenseClassifier(ClassifierPlugin):
#     default_config = {"host": "localhost", "port": 8500, "threshold": 0.5}

#     default_version = 0.2

#     def __init__(self, **kwargs):
#         super(KaggleDenseClassifier, self).__init__(**kwargs)
#         self._channel = grpc.insecure_channel(f'{self.config["host"]}:{self.config["port"]}')
#         self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
#         request = predict_pb2.PredictRequest()
#         request.model_spec.name = "kaggle_iart_densenet_201_export"
#         request.model_spec.signature_name = "serving_default"

#     def call(self, entries):
#         result_entries = []
#         result_annotations = []
#         for entry in entries:

#             logging.info(f'{self.name}: {entry["id"]}')
#             entry_annotation = []
#             with open(entry["path"], "rb") as f:
#                 # build request
#                 request = predict_pb2.PredictRequest()
#                 request.model_spec.name = "kaggle_iart_densenet_201_export"
#                 request.model_spec.signature_name = "serving_default"
#                 request.inputs["image_data"].CopyFrom(tf.make_tensor_proto(f.read(), shape=[1]))
#                 result_future = self._stub.Predict.future(request, 5.0)

#                 # get results
#                 results = result_future.result()

#                 # decode results
#                 artist_name = np.array(results.outputs["artist_name"].string_val)
#                 artist = np.array(results.outputs["artist"].int64_val)
#                 artist_probabilities = np.array(results.outputs["artist_probabilities"].float_val)
#                 genre_name = np.array(results.outputs["genre_name"].string_val)
#                 genre = np.array(results.outputs["genre"].int64_val)
#                 genre_probabilities = np.array(results.outputs["genre_probabilities"].float_val)
#                 style_name = np.array(results.outputs["style_name"].string_val)
#                 style = np.array(results.outputs["style"].int64_val)
#                 style_probabilities = np.array(results.outputs["style_probabilities"].float_val)

#                 if genre_probabilities[genre] > self.config["threshold"]:
#                     entry_annotation.append(
#                         {
#                             "type": "genre",
#                             "name": genre_name.item().decode("utf-8"),
#                             "value": genre_probabilities[genre].item(),
#                         }
#                     )

#                 if style_probabilities[style] > self.config["threshold"]:
#                     entry_annotation.append(
#                         {
#                             "type": "style",
#                             "name": style_name.item().decode("utf-8"),
#                             "value": style_probabilities[style].item(),
#                         }
#                     )

#             result_annotations.append(entry_annotation)
#             result_entries.append(entry)

#         return PluginResult(self, result_entries, result_annotations)


# #
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='')
#
#     parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
#     parser.add_argument('-p', '--port', default=8501, type=int, help='verbose output')
#     parser.add_argument('-c', '--host', default='localhost', help='verbose output')
#     parser.add_argument('-i', '--image', help='verbose output')
#     args = parser.parse_args()
#     return args
#
#
# def main():
#     args = parse_args()
#
#     channel = grpc.insecure_channel(f'{args.host}:{args.port}')
#     stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'kaggle_iart_densenet_201_export'
#     request.model_spec.signature_name = 'serving_default'
#     with open(args.image, 'rb') as f:
#         image = f.read()
#     request.inputs['image_data'].CopyFrom(tf.contrib.util.make_tensor_proto(image, shape=[1]))
#
#     result_future = stub.Predict.future(request, 5.0)  # 5 seconds
#     results = result_future.result()
#     print(results.outputs)
#     print(np.array(results.outputs['artist_name'].string_val))
#     print(np.array(results.outputs['artist'].int64_val))
#     print(np.array(results.outputs['artist_probabilities'].float_val))
#     print(np.array(results.outputs['genre_name'].string_val))
#     print(np.array(results.outputs['genre'].int64_val))
#     print(np.array(results.outputs['genre_probabilities'].float_val))
#     print(np.array(results.outputs['style_name'].string_val))
#     print(np.array(results.outputs['style'].int64_val))
#     print(np.array(results.outputs['style_probabilities'].float_val))
#     return 0
#
#
# if __name__ == '__main__':
#     sys.exit(main())
