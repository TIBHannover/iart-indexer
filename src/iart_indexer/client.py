import functools
import json
import logging
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os
import re
import sys
import struct
import time
import uuid

import grpc
import imageio
import random
import msgpack

from iart_indexer import indexer_pb2, indexer_pb2_grpc
from iart_indexer.utils import image_resize


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
            "origin": [],
        }
        for path in paths
    ]

    return entries


def list_jsonl(paths, image_paths=None):
    entries = []
    with open(paths, "r") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)

            if "path" not in entry:
                entry["path"] = os.path.join(image_paths, entry["id"][0:2], entry["id"][2:4], f"{entry['id']}.jpg")
            else:
                if not os.path.isabs(entry["path"]):
                    entry["path"] = os.path.join(image_paths, entry["path"])
                if not os.path.exists(entry["path"]):
                    entry["path"] = os.path.join(image_paths, entry["id"][0:2], entry["id"][2:4], f"{entry['id']}.jpg")

            if os.path.exists(entry["path"]):
                entries.append(entry)
                # print(entries)
                # break

            logging.info(f"{len(entries)}")

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


def copy_image(entry, image_output, image_paths=None, resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}]):
    path = entry["path"]
    copy_result = copy_image_hash(path, image_output, entry["id"], resolutions)
    if copy_result is not None:
        hash_value, path = copy_result
        entry.update({"path": path})
        return entry
    return None


def copy_images(
    entries, image_output, image_paths=None, resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}]
):
    entires_result = []
    with mp.Pool(8) as p:
        for entry in p.imap(
            functools.partial(copy_image, image_output=image_output, image_paths=image_paths, resolutions=resolutions),
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

    def copy_images(self, paths, image_paths=None, image_output=None):
        if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == ".jsonl":
            logging.info("Read json file")
            entries = list_jsonl(paths, image_paths)
        else:
            entries = list_images(paths)

        logging.info(f"Client: Copying {len(entries)} images to {image_output}")
        if image_output:
            entries = copy_images(entries, image_paths=image_paths, image_output=image_output)

        return entries

    def indexing(self, paths, image_paths=None, plugins: list = None):
        if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == ".jsonl":
            entries = list_jsonl(paths, image_paths)
        else:
            entries = list_images(paths)

        logging.info(f"Client: Start indexing {len(entries)} images")

        def entry_generator(entries, blacklist):

            for entry in entries:
                if blacklist is not None and entry["id"] in blacklist:
                    continue

                request = indexer_pb2.IndexingRequest()
                request_image = request.image

                request_image.id = entry["id"]

                for k, v in entry["meta"].items():

                    if isinstance(v, (list, set)):
                        for v_1 in v:
                            meta_field = request_image.meta.add()
                            meta_field.key = k
                            if isinstance(v_1, int):
                                meta_field.int_val = v_1
                            if isinstance(v_1, float):
                                meta_field.float_val = v_1
                            if isinstance(v_1, str):
                                meta_field.string_val = v_1
                    else:
                        meta_field = request_image.meta.add()
                        meta_field.key = k
                        if isinstance(v, int):
                            meta_field.int_val = v
                        if isinstance(v, float):
                            meta_field.float_val = v
                        if isinstance(v, str):
                            meta_field.string_val = v

                if "origin" in entry:

                    for k, v in entry["origin"].items():

                        if isinstance(v, (list, set)):
                            for v_1 in v:
                                origin_field = request_image.origin.add()
                                origin_field.key = k
                                if isinstance(v_1, int):
                                    origin_field.int_val = v_1
                                if isinstance(v_1, float):
                                    origin_field.float_val = v_1
                                if isinstance(v_1, str):
                                    origin_field.string_val = v_1
                        else:
                            origin_field = request_image.origin.add()
                            origin_field.key = k
                            if isinstance(v, int):
                                origin_field.int_val = v
                            if isinstance(v, float):
                                origin_field.float_val = v
                            if isinstance(v, str):
                                origin_field.string_val = v

                if "collection" in entry:
                    collection = request_image.collection
                    if "id" in entry["collection"]:
                        collection.id = entry["collection"]["id"]
                    if "name" in entry["collection"]:
                        collection.name = entry["collection"]["name"]
                    if "is_public" in entry["collection"]:
                        collection.is_public = entry["collection"]["is_public"]
                # print(request_image)
                request_image.encoded = open(entry["path"], "rb").read()
                yield request
            # request_image.path = image.encode()

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 2 ** 31 - 1),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)

        time_start = time.time()
        # gen_iter = entry_generator(entries)
        count = 0

        blacklist = set()
        try_count = 20
        while try_count > 0:
            try:
                gen_iter = entry_generator(entries, blacklist)
                # print('#####')
                # for x in gen_iter:
                #     print(x.image.id)
                #     blacklist.add(x.image.id)
                #     raise
                for i, entry in enumerate(stub.indexing(gen_iter)):
                    # for i, entry in enumerate(entry_generator(entries)):
                    blacklist.add(entry.id)
                    count += 1
                    if count % 1000 == 0:
                        speed = count / (time.time() - time_start)
                        logging.info(f"Client: Indexing {count}/{len(entries)} speed:{speed}")
                try_count = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(e)
                try_count -= 1

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

    def build_suggester(self, field_name=None):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)

        request = indexer_pb2.SuggesterRequest()
        if field_name is None:
            field_name = [
                "meta.title",
                "meta.artist_name",
                "meta.location",
                "meta.institution",
                "meta.object_type",
                "meta.medium",
                "origin.name",
                "classifier.*",
            ]

        request.field_names.extend(field_name)
        response = stub.build_suggester(request)

        return response.id

    def search(self, query):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.SearchRequest()

        print("BUILD QUERY")
        for q in query["queries"]:
            print(q)

            if "field" in q and q["field"] is not None:
                type_req = q["field"]
                if not isinstance(type_req, str):
                    return JsonResponse({"status": "error"})

                term = request.terms.add()
                term.text.query = q["query"]
                term.text.field = q["field"]
                term.text.flag = q["flag"]

            elif "query" in q and q["query"] is not None:
                term = request.terms.add()
                term.text.query = q["query"]

            if "reference" in q and q["reference"] is not None:
                request.sorting = "feature"

                term = request.terms.add()
                # TODO use a database for this case
                if os.path.exists(q["reference"]):
                    term.feature.image.encoded = open(q["reference"], "rb").read()
                else:
                    term.feature.image.id = q["reference"]

                if "features" in q:
                    plugins = q["features"]
                    if not isinstance(q["features"], (list, set)):
                        plugins = [q["features"]]
                    for p in plugins:
                        for k, v in p.items():
                            plugins = term.feature.plugins.add()
                            plugins.name = k.lower()
                            plugins.weight = v

        if "sorting" in query and query["sorting"] == "random":
            request.sorting = "random"

        if "mapping" in query and query["mapping"] == "umap":
            request.mapping = "umap"

        print(request)

        response = stub.search(request)

        status_request = indexer_pb2.ListSearchResultRequest(id=response.id)
        for x in range(600):
            try:
                response = stub.list_search_result(status_request)
                return response
            except grpc.RpcError as e:

                # search is still running
                if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                    pass  # {"status": "running"}
            return
            time.sleep(0.01)
        return {"error"}

    def get(self, id):
        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.GetRequest(id=id)
        response = stub.get(request)

        return response

    def build_indexer(self, rebuild=False):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.BuildIndexerRequest()
        request.rebuild = rebuild

        logging.info(f"rebuild {request.rebuild}")
        response = stub.build_indexer(request)
        print(response)
        return response

    def build_feature_cache(self):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.BuildFeatureCacheRequest()
        response = stub.build_feature_cache(request)

        return response

    def dump(self, output_path, origin):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.DumpRequest(origin=origin)
        with open(output_path, "wb") as f:
            for i, x in enumerate(stub.dump(request)):

                f.write(x.entry)
                if i % 1000 == 0:
                    print(i)

    def load(self, input_path):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)

        def entry_generator(path, blacklist=None):
            with open(path, "rb") as f:
                unpacker = msgpack.Unpacker(f)
                for entry in unpacker:
                    if blacklist is not None and entry["id"] in blacklist:
                        continue
                    yield indexer_pb2.LoadRequest(entry=msgpack.packb(entry))

        time_start = time.time()
        # gen_iter = entry_generator(entries)
        count = 0
        blacklist = set()
        try_count = 20
        while try_count > 0:
            try:
                for i, entry in enumerate(stub.load(entry_generator(input_path, blacklist))):
                    # for i, entry in enumerate(entry_generator(entries)):
                    blacklist.add(entry.id)
                    count += 1
                    if count % 1000 == 0:
                        speed = count / (time.time() - time_start)
                        logging.info(f"Client: Load {count} speed:{speed}")
                try_count = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(e)
                try_count -= 1

    def aggregate(self, part, type, field_name, size):

        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = indexer_pb2_grpc.IndexerStub(channel)
        request = indexer_pb2.AggregateRequest(type=type, part=part, field_name=field_name, size=size)
        response = stub.aggregate(request)

        return response
