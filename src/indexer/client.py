import re
import os
import time
import uuid
import imageio
import struct
import logging
import functools
import multiprocessing as mp

import grpc
import json

from indexer import indexer_pb2
from indexer import indexer_pb2_grpc

from indexer.utils import image_resize


def list_images(paths, name_as_hash=False):
    if not isinstance(paths, (list, set)):

        if os.path.isdir(paths):
            file_paths = []
            for root, dirs, files in os.walk(paths):
                for f in files:
                    file_path = os.path.abspath(os.path.join(root, f))
                    # TODO filter
                    file_paths.append(file_path)

            paths = file_paths
        else:
            paths = [os.path.abspath(paths)]

    entries = [
        {
            "id": os.path.splitext(os.path.basename(path))[0] if name_as_hash else uuid.uuid4().hex,
            "filename": os.path.basename(path),
            "path": os.path.abspath(path),
            "meta": [],
        }
        for path in paths
    ]

    return entries


def list_jsonl(paths):
    entries = []
    with open(paths, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

    return entries


def copy_image_hash(image_path, image_output, hash_value=None, resolutions=[{"min_dim": -1, "suffix": ""}]):
    try:
        if hash_value is None:
            hash_value = uuid.uuid4().hex

        image_output_dir = os.path.join(image_output, hash_value[0:2], hash_value[2:4])
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        image = imageio.imread(image_path)

        for res in resolutions:
            if "min_dim" in res:
                new_image = image_resize(image, min_dim=res["min_dim"])

                image_output_file = os.path.join(image_output_dir, f"{hash_value}{res['suffix']}.jpg")
            else:
                new_image = image
                image_output_file = os.path.join(image_output_dir, f"{hash_value}{res['suffix']}.jpg")

            imageio.imwrite(image_output_file, new_image)

        image_output_file = os.path.abspath(os.path.join(image_output_dir, f"{hash_value}.jpg"))
        return hash, image_output_file
    except ValueError as e:
        return None
    except struct.error as e:
        return None


def copy_image(entry, image_output, image_input=None, resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}]):
    if image_input is not None:
        path = os.path.join(image_input, os.path.basename(entry["path"]))

    else:
        path = entry["path"]
    copy_result = copy_image_hash(path, image_output, entry["id"], resolutions)
    if copy_result is not None:
        hash_value, path = copy_result
        entry.update({"path": path})
        return entry
    return None


def copy_images(
    entries, image_output, image_input=None, resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}]
):
    entires_result = []
    with mp.Pool(8) as p:
        for entry in p.imap(
            functools.partial(copy_image, image_output=image_output, image_input=image_input, resolutions=resolutions),
            entries,
        ):
            if entry is not None:
                entires_result.append(entry)

    return entires_result


def split_batch(entries, batch_size=512):
    if batch_size < 1:
        return [entries]

    return [entries[x * batch_size : (x + 1) * batch_size] for x in range(len(entries) // batch_size + 1)]


class Client:
    def __init__(self, config):
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 50051)

    def plugin_list(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = indexer_pb2_grpc.IndexerStub(channel)
        response = stub.list_plugins(indexer_pb2.ListPluginsRequest())
        result = {}
        for plugin in response.plugins:
            if plugin.type not in result:
                result[plugin.type] = []

            result[plugin.type].append(plugin.name)

        return result

    def copy_images(self, paths, image_input=None, image_output=None):
        if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == ".jsonl":
            entries = list_jsonl(paths)
        else:
            entries = list_images(paths)

        logging.info(f"Client: Copying {len(entries)} images to {image_output}")
        if image_output:
            entries = copy_images(entries, image_input=image_input, image_output=image_output)

        return entries

    def indexing(self, paths, batch_size: int = 512, plugins: list = None):
        if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == ".jsonl":
            entries = list_jsonl(paths)
        else:
            entries = list_images(paths)

        logging.info(f"Client: Start indexing {len(entries)} images")

        entries_list = split_batch(entries, batch_size)

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)

        count = 0
        for entries_split in entries_list:
            request = indexer_pb2.IndexingRequest()
            request.update_database = True
            if plugins is None:
                # TODO
                pass
            else:
                for plugin in plugins:
                    request_plugin = request.plugins.add()
                    request_plugin.name = plugin
            for entry in entries_split:
                request_image = request.images.add()
                request_image.id = entry["id"]
                # {"id": "d09d10a5b6474997ae2580086b2e4666", "meta": {"title": "Altes Rathaus", "year_min": 1267, "yaer_max": 1267, "location": "Aachen", "institution": "Rathaus"}, "path": "/home/matthias/projects/iart/web/media/d0/9d/d09d10a5b6474997ae2580086b2e4666.jpg", "filename": "130.jpg"}
                # {"id": "8cfc09f13f0b45c8a56dcae17c33ed10", "meta": {"title": "Kirche Sankt Justinus (Mittelschiffarkaden)", "year_min": 925, "yaer_max": 950, "location": "H\u00f6chst (Frankfurt)", "institution": "Kirche Sankt Justinus"}, "path": "/home/matthias/projects/iart/web/media/8c/fc/8cfc09f13f0b45c8a56dcae17c33ed10.jpg", "filename": "136.jpg"}

                for k, v in entry["meta"].items():

                    meta_field = request_image.meta.add()
                    meta_field.key = k
                    if isinstance(v, int):
                        meta_field.int_val = v
                    if isinstance(v, float):
                        meta_field.float_val = v
                    if isinstance(v, str):
                        meta_field.string_val = v

                request_image.encoded = open(entry["path"], "rb").read()
            # request_image.path = image.encode()
            response = stub.indexing(request)

            # TODO timeout
            while True:
                status = self.status(response.id)
                if status == "done":
                    break
                print(status)
                time.sleep(1)
            count += len(entries_split)
            logging.info(f"Client: Indexing {count}/{len(entries)} images")

    def status(self, job_id):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.StatusRequest()
        request.id = job_id
        response = stub.status(request)

        return response.status
