import uuid
import logging
import numpy as np

from typing import Dict


from iart_indexer import indexer_pb2, data_pb2
from iart_indexer.plugins import ComputePlugin, ComputePluginManager


default_config = {
    "multicrop": True,
    "max_dim": None,
    "min_dim": 224,
}


default_config = {
    "multicrop": True,
    "max_dim": None,
    "min_dim": 224,
}

default_parameters = {"crop_size": [224, 224], "aggregation": "softmax"}


@ComputePluginManager.export("ClipClassification")
class ClipClassification(ComputePlugin, config=default_config, parameters=default_parameters, version="0.4"):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_dim = self.config["max_dim"]
        self.min_dim = self.config["min_dim"]

    def call(self, analyse_request: indexer_pb2.AnalyseRequest):
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        inputs, parameters = self.map_analyser_request_to_dict(analyse_request)

        image_embedding_request = self.map_dict_to_analyser_request({"image": inputs["image"]}, parameters)

        image_embedding_result = self.inference_server(
            self.compute_plugin_manager, "ClipImageEmbeddingFeature", image_embedding_request
        )

        text_embedding_request = self.map_dict_to_analyser_request({"text": inputs["text"]}, parameters)

        text_embedding_result = self.inference_server(
            self.compute_plugin_manager, "ClipTextEmbeddingFeature", text_embedding_request
        )

        text_embeddings = []
        for t in text_embedding_result.results:
            text_embeddings.append(np.asarray(t.result.feature.feature).reshape(t.result.feature.shape))

        text_embeddings = np.stack(text_embeddings, 0)

        image_embeddings = []
        for i in image_embedding_result.results:
            image_embeddings.append(np.asarray(i.result.feature.feature).reshape(i.result.feature.shape))

        image_embeddings = np.stack(image_embeddings, 0)

        if parameters.get("aggregation").lower() == "softmax":
            text_probs = torch.nn.functional.softmax(
                torch.from_numpy(100.0 * image_embeddings @ text_embeddings.T), dim=-1
            )

        if parameters.get("aggregation").lower() == "dot":
            text_probs = image_embeddings @ text_embeddings.T

        if parameters.get("aggregation").lower() == "cosine":
            text_probs = cosine_similarity(image_embeddings, text_embeddings)

        result = indexer_pb2.AnalyseReply()

        for i, _ in enumerate(inputs["image"]):
            concepts = []

            for j, concept in enumerate(inputs["text"]):
                concepts.append(data_pb2.Concept(concept=concept["content"], prob=text_probs[i, j].item()))

            data = data_pb2.PluginData(
                id=uuid.uuid4().hex,
                name="clip_embedding",
                classifier=data_pb2.ClassifierResult(concepts=concepts),
            )

            result.results.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type="",
                    version="",
                    result=data,
                )
            )

        return result
