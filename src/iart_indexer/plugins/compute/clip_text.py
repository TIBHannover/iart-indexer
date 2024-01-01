import numpy as np
import uuid
from iart_indexer import indexer_pb2, data_pb2
from iart_indexer.plugins import ComputePlugin, ComputePluginManager, ComputePluginResult
from iart_indexer.utils import image_from_proto, image_resize, image_crop


import logging

from typing import Union, List, Dict


default_config = {}


default_parameters = {}


@ComputePluginManager.export("ClipTextEmbeddingFeature")
class ClipTextEmbeddingFeature(ComputePlugin, config=default_config, parameters=default_parameters, version="0.4"):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.config.get("model", "xlm-roberta-base-ViT-B-32")
        self.pretrained = self.config.get("pretrained", "laion5b_s13b_b90k")
        self.model = None

    def call(self, analyse_request: indexer_pb2.AnalyseRequest):
        from sklearn.preprocessing import normalize
        import imageio.v3 as iio
        import torch
        import open_clip

        inputs, parameters = self.map_analyser_request_to_dict(analyse_request)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.error(f"DEVICE {device}")
        if self.model is None:
            logging.error(f"LOAD {device}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                cache_dir="/models",
                device=device,
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model = model
        # text = self.preprocess(parameters["search_term"])

        result = indexer_pb2.AnalyseReply()
        logging.error(len(inputs["text"]))
        for entry in inputs["text"]:
            logging.error(len(inputs["text"]))
            with torch.no_grad(), torch.cuda.amp.autocast():
                text = self.tokenizer(entry["content"])
                output = self.model.encode_text(text.to(device))
            output = output / np.linalg.norm(np.asarray(output))
            output = output.flatten()
            data = data_pb2.PluginData(
                id=uuid.uuid4().hex,
                name="clip_embedding",
                feature=data_pb2.Feature(type="clip_embedding", shape=output.shape, feature=output.tolist()),
            )

            result.results.append(
                indexer_pb2.PluginResult(
                    plugin=self.name,
                    type="",
                    version=self.version,
                    result=data,
                )
            )
        return result
