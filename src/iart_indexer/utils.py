import argparse
import os
import pickle
import re
import struct
import sys
import uuid

import imageio
import numpy as np
import PIL
import numbers

from iart_indexer import indexer_pb2


def get_features_from_db_entry(entry):
    data_dict = {"id": entry["id"], "feature": []}
    # TODO
    if "feature" not in entry:
        return data_dict
    for feature in entry["feature"]:
        for annotation in feature["annotations"]:
            if "value" in annotation:
                value = annotation["value"]
            if "val_64" in annotation:
                value = annotation["val_64"]
            if "val_128" in annotation:
                value = annotation["val_128"]
            if "val_256" in annotation:
                value = annotation["val_256"]
            data_dict["feature"].append(
                {"plugin": feature["plugin"], "type": annotation["type"], "version": feature["version"], "value": value}
            )

    return data_dict


def get_classifier_from_db_entry(entry):

    if "classifier" not in entry:
        return {"id": entry["id"], "classifier": []}
    data_dict = {"id": entry["id"], "classifier": entry["classifier"]}
    return data_dict


# TODO
def get_features_from_db_plugins(entry):
    data_dict = {"id": entry["id"], "feature": []}
    # TODO
    if "feature" not in entry:
        return data_dict
    for feature in entry["feature"]:
        for annotation in feature["annotations"]:
            if "value" in annotation:
                value = annotation["value"]
            if "val_64" in annotation:
                value = annotation["val_64"]
            if "val_128" in annotation:
                value = annotation["val_128"]
            if "val_256" in annotation:
                value = annotation["val_256"]
            data_dict["feature"].append({"plugin": feature["plugin"], "type": annotation["type"], "value": value})

    return data_dict


def image_normalize(image):
    if len(image.shape) == 2:

        return np.stack([image] * 3, -1)

    if len(image.shape) == 3:
        if image.shape[-1] == 4:
            return image[..., 0:3]
        if image.shape[-1] == 1:
            return np.concatenate([image] * 3, -1)

    if len(image.shape) == 4:
        return image_normalize(image[0, ...])

    return image


def image_crop(image, size=None):

    image = PIL.Image.fromarray(image)
    width, height = image.size  # Get dimensions

    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    return np.array(image)


def image_from_proto(image_proto):
    image_field = image_proto.WhichOneof("image")
    if image_field == "path":
        image = imageio.imread(image_proto.path)
    if image_field == "encoded":
        image = imageio.imread(image_proto.encoded)

    # if len(image.shape) == 2:
    #     image = np.stack([image] * 3, -1)
    return image_normalize(image)


def image_resize(image, max_dim=None, min_dim=None, size=None):
    if max_dim is not None:
        shape = np.asarray(image.shape[:2], dtype=np.float32)

        long_dim = max(shape)
        scale = min(1, max_dim / long_dim)
        new_shape = np.asarray(shape * scale, dtype=np.int32)

    elif min_dim is not None:
        shape = np.asarray(image.shape[:2], dtype=np.float32)

        short_dim = min(shape)
        scale = min(1, min_dim / short_dim)
        new_shape = np.asarray(shape * scale, dtype=np.int32)
    elif size is not None:
        new_shape = size
    else:
        return image
    img = PIL.Image.fromarray(image)
    img = img.resize(size=new_shape[::-1])
    return np.array(img)


def prediction_to_proto(prediction):
    result = indexer_pb2.PluginResult()


def prediction_from_proto(proto):
    pass


def meta_from_proto(proto):
    result_dict = {}
    for m in proto:
        field = m.WhichOneof("value")
        if field == "string_val":
            result_dict[m.key] = m.string_val
        if field == "int_val":
            result_dict[m.key] = m.int_val
        if field == "float_val":
            result_dict[m.key] = m.float_val
    return result_dict


def meta_to_proto(proto, data):

    for k, v in data.items():
        meta = proto.add()
        if isinstance(v, numbers.Integral):
            meta.int_val = v
            meta.key = k
        elif isinstance(v, numbers.Real):
            meta.float_val = v
            meta.key = k
        elif isinstance(v, str):
            meta.string_val = v
            meta.key = k
    return proto


def classifier_from_proto(proto):
    result_list = []
    for c in proto:
        result_list.append(
            {"plugin": c.plugin, "annotations": [{"name": a.concept, "value": a.prob} for a in c.concepts]}
        )
    return result_list


def classifier_to_proto(proto, data):
    for c in data:
        classifier = proto.add()
        classifier.plugin = c["plugin"]
        for a in c["annotations"]:
            concept = classifier.concepts.add()
            concept.concept = a["name"]
            concept.type = a["type"]
            concept.prob = a["value"]
    return proto


def feature_from_proto(proto):
    result_list = []
    for f in proto:
        result_list.append({"plugin": f.plugin, "annotations": [{"feature": list(f.feature), "type": f.type}]})
    return result_list


def feature_to_proto(proto, data):
    for f in data:
        feature = proto.add()
        feature.plugin = f["plugin"]
        # TODO support list here
        for a in f["annotations"]:
            if "val_64" in a:
                feature.feature.extend(a["val_64"])

            if "val_128" in a:
                feature.feature.extend(a["val_128"])

            if "val_256" in a:
                feature.feature.extend(a["val_256"])
            feature.type = a["type"]
    return proto


def suggestions_from_proto(proto):
    result_list = []
    for g in proto.groups:
        group_dict = {"group": g.group, "suggestions": list(g.suggestions)}

        result_list.append(group_dict)
    return result_list


def convert_name(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def numpy_to_proto(proto, name, value, frame_number, timestamp):
    ENCODE_DTYPE_LUP = {
        np.dtype("float16"): indexer_pb2.DT_HALF,
        np.dtype("float32"): indexer_pb2.DT_FLOAT,
        np.dtype("float64"): indexer_pb2.DT_DOUBLE,
        np.dtype("int32"): indexer_pb2.DT_INT32,
        np.dtype("int64"): indexer_pb2.DT_INT64,
        np.dtype("uint8"): indexer_pb2.DT_UINT8,
        np.dtype("uint16"): indexer_pb2.DT_UINT16,
        np.dtype("uint32"): indexer_pb2.DT_UINT32,
        np.dtype("uint64"): indexer_pb2.DT_UINT64,
        np.dtype("int16"): indexer_pb2.DT_INT16,
        np.dtype("int8"): indexer_pb2.DT_INT8,
        np.dtype("object"): indexer_pb2.DT_STRING,
        np.dtype("bool"): indexer_pb2.DT_BOOL,
    }
    # proto = indexer_pb2.VideoProto

    proto.name = name
    proto.frame_number = frame_number
    proto.timestamp = timestamp

    proto.dtype = ENCODE_DTYPE_LUP[value.dtype]

    proto.proto_content = pickle.dumps(value)
    proto.shape.extend(value.shape)
    return proto


def numpy_from_proto(proto):
    DECODE_DTYPE_LUP = {
        indexer_pb2.DT_HALF: np.float16,
        indexer_pb2.DT_FLOAT: np.float32,
        indexer_pb2.DT_DOUBLE: np.float64,
        indexer_pb2.DT_INT32: np.int32,
        indexer_pb2.DT_UINT8: np.uint8,
        indexer_pb2.DT_UINT16: np.uint16,
        indexer_pb2.DT_UINT32: np.uint32,
        indexer_pb2.DT_UINT64: np.uint64,
        indexer_pb2.DT_INT16: np.int16,
        indexer_pb2.DT_INT8: np.int8,
        indexer_pb2.DT_STRING: np.object,
        indexer_pb2.DT_INT64: np.int64,
        indexer_pb2.DT_BOOL: np.bool,
    }

    content = pickle.loads(proto.proto_content)
    # content = np.frombuffer(proto.proto_content, dtype=DECODE_DTYPE_LUP[proto.dtype])
    content.reshape(proto.shape)
    return proto.name, content, proto.frame_number, proto.timestamp
