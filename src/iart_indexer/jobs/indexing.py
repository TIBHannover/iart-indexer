import logging
import imageio
import traceback

from google.protobuf.json_format import MessageToJson, MessageToDict, ParseDict

from iart_indexer import indexer_pb2, indexer_pb2_grpc
from iart_indexer.plugins import ClassifierPluginManager, FeaturePluginManager
from iart_indexer.utils import image_normalize


class IndexingJob:
    def __init__(self, config=None):
        if config is not None:
            self.init_worker(config)

    @classmethod
    def init_worker(cls, config):
        logging.info("INIT")

        feature_manager = FeaturePluginManager(configs=config.get("features", []))
        feature_manager.find()

        setattr(cls, "feature_manager", feature_manager)

        classifier_manager = ClassifierPluginManager(configs=config.get("classifiers", []))
        classifier_manager.find()

        setattr(cls, "classifier_manager", classifier_manager)

    @classmethod
    def __call__(cls, entry):
        logging.info("CALL")

        classifier_manager = getattr(cls, "classifier_manager")
        feature_manager = getattr(cls, "feature_manager")
        try:
            image = imageio.imread(entry["image_data"])
            image = image_normalize(image)
        except Exception as e:
            logging.error(traceback.format_exc())
            return "error", {"id": entry["id"]}
        plugins = []
        for c in entry["cache"]["classifier"]:
            plugins.append({"plugin": c["plugin"], "version": c["version"]})

        classifications = list(classifier_manager.run([image], [plugins]))[0]

        plugins = []
        for c in entry["cache"]["feature"]:
            plugins.append({"plugin": c["plugin"], "version": c["version"]})
        features = list(feature_manager.run([image], [plugins]))[0]

        doc = {"id": entry["id"], "meta": entry["meta"], "origin": entry["origin"], "collection": entry["collection"]}

        annotations = []
        for f in features["plugins"]:

            for anno in f._annotations:

                for result in anno:
                    plugin_annotations = []

                    binary_vec = result.feature.binary
                    feature_vec = list(result.feature.feature)

                    plugin_annotations.append(
                        {
                            "type": result.feature.type,
                            "value": feature_vec,
                        }
                    )

                    feature_result = {
                        "plugin": result.plugin,
                        "version": result.version,
                        "annotations": plugin_annotations,
                    }
                    annotations.append(feature_result)

        if len(annotations) > 0:
            doc["feature"] = annotations

        annotations = []
        for c in classifications["plugins"]:

            for anno in c._annotations:

                for result in anno:
                    plugin_annotations = []
                    for concept in result.classifier.concepts:
                        plugin_annotations.append(
                            {"name": concept.concept, "type": concept.type, "value": concept.prob}
                        )

                    classifier_result = {
                        "plugin": result.plugin,
                        "version": result.version,
                        "annotations": plugin_annotations,
                    }
                    annotations.append(classifier_result)

        if len(annotations) > 0:
            doc["classifier"] = annotations

        # copy predictions from cache
        for exist_c in entry["cache"]["classifier"]:
            if "classifier" not in doc:
                doc["classifier"] = []

            founded = False
            for computed_c in doc["classifier"]:
                if computed_c["plugin"] == exist_c["plugin"] and version.parse(
                    str(computed_c["version"])
                ) > version.parse(str(exist_c["version"])):
                    founded = True

            if not founded:
                doc["classifier"].append(exist_c)

        for exist_f in entry["cache"]["feature"]:
            if "feature" not in doc:
                doc["feature"] = []
            exist_f_version = version.parse(str(exist_f["version"]))

            founded = False
            for computed_f in doc["feature"]:
                computed_f_version = version.parse(str(computed_f["version"]))
                if computed_f["plugin"] == exist_f["plugin"] and computed_f_version >= exist_f_version:
                    founded = True
            if not founded:
                exist_f = {
                    "plugin": exist_f["plugin"],
                    "version": exist_f["version"],
                    "annotations": [
                        {
                            "type": exist_f["type"],
                            "value": exist_f["value"],
                        }
                    ],
                }
                doc["feature"].append(exist_f)

        return "ok", doc
