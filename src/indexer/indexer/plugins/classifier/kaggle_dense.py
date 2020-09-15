from indexer.indexer.plugins import ClassifierPlugin
from indexer.indexer.plugins import ClassifierPluginManager
from indexer.indexer.plugins import PluginResult

import os
import sys
import re
import argparse
import threading
import grpc
import numpy as np

# import tensorflow as tf

# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc

import logging


@ClassifierPluginManager.export("KaggleDenseClassifier")
class KaggleDenseClassifier(ClassifierPlugin):
    default_config = {"host": "localhost", "port": 8500, "threshold": 0.5}

    default_version = 0.2

    def __init__(self, **kwargs):
        super(KaggleDenseClassifier, self).__init__(**kwargs)
        self._channel = grpc.insecure_channel(f'{self.config["host"]}:{self.config["port"]}')
        self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = "kaggle_iart_densenet_201_export"
        request.model_spec.signature_name = "serving_default"

    def call(self, entries):
        result_entries = []
        result_annotations = []
        for entry in entries:

            logging.info(f'{self.name}: {entry["id"]}')
            entry_annotation = []
            with open(entry["path"], "rb") as f:
                # build request
                request = predict_pb2.PredictRequest()
                request.model_spec.name = "kaggle_iart_densenet_201_export"
                request.model_spec.signature_name = "serving_default"
                request.inputs["image_data"].CopyFrom(tf.make_tensor_proto(f.read(), shape=[1]))
                result_future = self._stub.Predict.future(request, 5.0)

                # get results
                results = result_future.result()

                # decode results
                artist_name = np.array(results.outputs["artist_name"].string_val)
                artist = np.array(results.outputs["artist"].int64_val)
                artist_probabilities = np.array(results.outputs["artist_probabilities"].float_val)
                genre_name = np.array(results.outputs["genre_name"].string_val)
                genre = np.array(results.outputs["genre"].int64_val)
                genre_probabilities = np.array(results.outputs["genre_probabilities"].float_val)
                style_name = np.array(results.outputs["style_name"].string_val)
                style = np.array(results.outputs["style"].int64_val)
                style_probabilities = np.array(results.outputs["style_probabilities"].float_val)

                if genre_probabilities[genre] > self.config["threshold"]:
                    entry_annotation.append(
                        {
                            "type": "genre",
                            "name": genre_name.item().decode("utf-8"),
                            "value": genre_probabilities[genre].item(),
                        }
                    )

                if style_probabilities[style] > self.config["threshold"]:
                    entry_annotation.append(
                        {
                            "type": "style",
                            "name": style_name.item().decode("utf-8"),
                            "value": style_probabilities[style].item(),
                        }
                    )

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)


#
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
