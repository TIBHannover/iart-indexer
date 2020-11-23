import math
import uuid

import numpy as np

from iart_indexer import indexer_pb2
from iart_indexer.plugins import MappingPlugin, MappingPluginManager, PluginResult
from iart_indexer.utils import image_from_proto, image_resize


@MappingPluginManager.export("FeatureCosineMapping")
class FeatureCosineMapping(MappingPlugin):
    default_config = {}

    default_version = 0.1

    def __init__(self, **kwargs):
        super(FeatureCosineMapping, self).__init__(**kwargs)

    def call(self, entries, query):

        if query is not None:
            new_entries = []
            for e in entries:
                score = 0
                for q_f in query:
                    for e_f in e["feature"]:
                        if q_f["plugin"] != e_f["plugin"]:
                            continue
                        if "val_64" in e_f["annotations"][0]:
                            a = e_f["annotations"][0]["val_64"]
                            b = q_f["annotations"][0]["val_64"]
                            score += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) * q_f["weight"]
                        if "val_128" in e_f["annotations"][0]:
                            a = e_f["annotations"][0]["val_128"]
                            b = q_f["annotations"][0]["val_128"]
                            score += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) * q_f["weight"]
                        if "val_256" in e_f["annotations"][0]:
                            a = e_f["annotations"][0]["val_256"]
                            b = q_f["annotations"][0]["val_256"]
                            score += np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) * q_f["weight"]

                new_entries.append((score, {**e, "coordinates": [score]}))

            new_entries = sorted(new_entries, key=lambda x: -x[0])
            entries = [x[1] for x in new_entries]

        return entries
