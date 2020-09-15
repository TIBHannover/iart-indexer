import os
import sys
import re
import argparse
import uuid
import imageio
import struct
import PIL


import numpy as np
from indexer import indexer_pb2
import pickle


def image_from_proto(image_proto):
    if image_proto.path is not None:
        image = imageio.imread(image_proto.path)
    if image_proto.data is not None:
        pass

    return image


def image_resize(image, max_dim=None, min_dim=None, size=None):
    if max_dim is not None:
        shape = np.asarray(image.shape[:-1], dtype=np.float32)

        long_dim = max(shape)
        scale = min(1, max_dim / long_dim)
        new_shape = np.asarray(shape * scale, dtype=np.int32)

    elif min_dim is not None:
        shape = np.asarray(image.shape[:-1], dtype=np.float32)

        short_dim = min(shape)
        scale = min(1, min_dim / short_dim)
        new_shape = np.asarray(shape * scale, dtype=np.int32)
    elif size is not None:
        new_shape = size
    else:
        return image
    print(f"Resize image: {image.shape} {new_shape}")

    img = PIL.Image.fromarray(image)
    img = img.resize(size=new_shape)
    return np.array(img)


def prediction_to_proto(prediction):
    result = indexer_pb2.PluginResult()


def prediction_from_proto(proto):
    pass


def convert_name(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def copy_image_hash(image_path, output_path, hash_value=None, resolutions=[-1]):
    try:
        if hash_value is None:
            hash_value = uuid.uuid4().hex

        output_dir = os.path.join(output_path, hash_value[0:2], hash_value[2:4])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = imageio.imread(image_path)

        for res in resolutions:
            if res > 0:
                scale_factor = max(image.shape[0] / float(res), image.shape[1] / float(res))
                new_image = skimage.transform.rescale(image, 1 / scale_factor)

                output_file = os.path.join(output_dir, f"{hash_value}_{res}.jpg")
            else:
                new_image = image
                output_file = os.path.join(output_dir, f"{hash_value}.jpg")

            imageio.imwrite(output_file, new_image)

        output_file = os.path.abspath(os.path.join(output_dir, f"{hash_value}.jpg"))
        return hash, output_file
    except ValueError as e:
        return None
    except struct.error as e:
        return None


def image_resolution(image_path):
    try:
        image = imageio.imread(image_path)

        return image.shape[0], image.shape[1]
    except ValueError as e:
        return None
    except struct.error as e:
        return None


def filename_without_ext(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


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
