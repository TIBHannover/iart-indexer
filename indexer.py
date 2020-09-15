import os
import sys
import re
import json
import argparse
import logging
import uuid
import imageio

import tensorflow as tf

from indexer.plugins import *
from indexer.config import IndexerConfig

from database.elasticsearch_database import ElasticSearchDatabase

from indexer.utils import copy_image_hash, filename_without_ext, image_resolution

import time
from concurrent import futures
import threading

import indexer_pb2
import indexer_pb2_grpc
import grpc

from indexer.plugins import FeaturePlugin, ClassifierPlugin

from indexer.utils import image_from_proto

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def parse_args():
    parser = argparse.ArgumentParser(description="Indexing a set of images")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    # parser.add_argument('-l', '--list', help='list all plugins')
    parser.add_argument("-p", "--path", help="path to image or folder to indexing")

    parser.add_argument("-b", "--batch", default=512, type=int, help="split images in batch")
    parser.add_argument("-o", "--output", help="copy image to new folder with hash id")
    parser.add_argument("-a", "--append", action="store_true", help="add all found documents to the index")
    parser.add_argument("-j", "--jsonl", help="add all found documents to the index")
    # parser.add_argument('-d', '--database', help='database type for index')
    parser.add_argument("-c", "--config", help="config path")
    parser.add_argument("-m", "--mode", choices=["local", "server"], default="local", help="verbose output")
    args = parser.parse_args()
    return args


def compute_plugins(args):
    logging.info("Start computing job")
    plugin_classes = args["plugin_classes"]
    images = args["images"]
    database = args["database"]
    if database is not None:
        existing_hash = [x["id"] for x in list(database.all())]
        for x in images:
            if x["id"] not in existing_hash:
                resolution = image_resolution(x["path"])
                if resolution is None:
                    continue
                database.insert_entry(
                    x["id"],
                    {
                        "id": x["id"],
                        "path": x["path"],
                        "filename": x["filename"],
                        "meta": x["meta"],
                        "image": {"height": resolution[0], "width": resolution[1]},
                    },
                )
    plugin_result_list = {}
    for plugin_class in plugin_classes:
        plugin = plugin_class()
        plugin_results = plugin(images)
        # # TODO entries_processed also contains the entries zip will be

        # for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):
        #     if entry["id"] not in plugin_result_list:
        #         plugin_result_list[entry["id"]] = {}
        #     if isinstance(plugin, ClassifierPlugin):
        #         if "classifier" not in plugin_result_list[i]:

        #             plugin_result_list[i]["classifier"] = []

        #         plugin_result_list[i]["classifier"].append(prediction)
        #     if isinstance(plugin, FeaturePlugin):
        #         if "feature" not in plugin_result_list[i]:

        #             plugin_result_list[i]["feature"] = []
        #         plugin_result_list[i]["feature"].append(prediction)
        # print(plugin_class)
        # print(plugin_results)

        if database is not None:
            update_database(database, plugin_results)

        print(plugin_results)

    logging.info(plugin_result_list)
    return plugin_result_list


class Commune(indexer_pb2_grpc.IndexerServicer):
    def __init__(self, feature_manager, classifier_manager):
        self.feature_manager = feature_manager
        self.classifier_manager = classifier_manager
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=1)
        self.futures = {}

    def list_plugins(self, request, context):
        reply = indexer_pb2.ListPluginsReply()

        for plugin_name, plugin_class in self.feature_manager.plugins().items():
            pluginInfo = reply.plugins.add()
            pluginInfo.name = plugin_name
            pluginInfo.type = "feature"

        for plugin_name, plugin_class in self.classifier_manager.plugins().items():
            pluginInfo = reply.plugins.add()
            pluginInfo.name = plugin_name
            pluginInfo.type = "classifier"

            # for setting in self.pluginManager.settings(pluginName):
            #     settingReply = pluginInfo.settings.add()
            #     settingReply.name = setting["name"]
            #     settingReply.default = setting["default"]
            #     settingReply.type = setting["type"]

        return reply

    def indexing(self, request, context):
        plugin_list = []
        plugin_name_list = [x.name.lower() for x in request.plugins]
        for plugin_name, plugin_class in self.feature_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue
            plugin_list.append(plugin_class)

        for plugin_name, plugin_class in self.classifier_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue
            plugin_list.append(plugin_class)
        database = None
        if request.update_database:
            database = ElasticSearchDatabase(config=None)

        variable = {
            "plugin_classes": plugin_list,
            "images": request.images,
            "database": database,
            "progress": 0,
            "status": 0,
            "result": "",
            "future": None,
        }

        future = self.thread_pool.submit(compute_plugins, variable)

        job_id = uuid.uuid4().hex
        variable["future"] = future
        self.futures[job_id] = variable

        # thread = threading.Thread(target=compute_plugins, args=(variable,))

        # self.threads[job_id] = (thread, variable)
        # thread.start()

        return indexer_pb2.IndexingReply(id=job_id)

    # def UpdateStatus(self, request, context):

    #     unique_id = request.unique_id
    #     if unique_id in self.threads:
    #         US = indexer_pb2.Status()
    #         US.status = self.threads[unique_id][1]["progress"]
    #         US.result.dtype = indexer_pb2.DT_STRING
    #         US.result.proto_content = codecs.encode(self.threads[unique_id][1]["result"], "utf-8")

    #         return US

    def status(self, request, context):

        if request.id in self.futures:
            job_data = self.futures[request.id]
            done = job_data["future"].done()
            if not done:
                return indexer_pb2.StatusReply(status="running")

            result = job_data["future"].result()
            print(result)
            return indexer_pb2.StatusReply(status="done")

            # for x in content:
            #     infoReply = GI.info.add()
            #     infoReply.dtype = indexer_pb2.DT_FLOAT
            #     infoReply.name = name
            #     print(x)
            #     infoReply.proto_content = x.tobytes()
            #     print(len(infoReply.proto_content))
            #     infoReply.shape.extend(x.shape)

        return indexer_pb2.StatusReply(status="error")


def update_database(database, plugin_results):
    for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):

        hash_id = entry["id"]

        database.update_plugin(
            hash_id,
            plugin_name=plugin_results._plugin.name,
            plugin_version=plugin_results._plugin.version,
            plugin_type=plugin_results._plugin.type,
            annotations=annotations,
        )


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


def copy_images(entries, output, resolutions=[500, 200, -1]):
    entires_result = []
    for i in range(len(entries)):

        entry = entries[i]
        print(entry["path"])
        copy_result = copy_image_hash(entry["path"], output, entry["id"], resolutions)
        if copy_result is not None:
            hash_value, path = copy_result
            entry.update({"path": path})
            entires_result.append(entry)

    return entires_result


def split_batch(entries, batch_size=512):
    if batch_size < 1:
        return [entries]

    return [entries[x * batch_size : (x + 1) * batch_size] for x in range(len(entries) // batch_size + 1)]


class Indexer:
    def __init__(self, config):
        pass

    def indexing_files(
        self, paths=None, images=None,
    ):
        pass


def indexing(paths, output, batch_size: int = 512, plugins: list = [], config: dict = {}):

    # TODO replace with abstract class
    if "db" in config:
        database = ElasticSearchDatabase(config=config["db"])

    # handel images or jsonl
    if paths is not None:
        if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == ".jsonl":
            entries = list_jsonl(paths)
        else:
            entries = list_images(paths)

        if output:
            entries = copy_images(entries, output)

    else:
        entries = list(database.all())

    # TODO scroll
    existing_hash = [x["id"] for x in list(database.all())]
    for x in entries:
        if x["id"] not in existing_hash:
            resolution = image_resolution(x["path"])
            if resolution is None:
                continue
            database.insert_entry(
                x["id"],
                {
                    "id": x["id"],
                    "path": x["path"],
                    "filename": x["filename"],
                    "meta": x["meta"],
                    "image": {"height": resolution[0], "width": resolution[1]},
                },
            )

    logging.info(f"Indexing {len(entries)} documents")

    entries_list = split_batch(entries, batch_size)

    feature_manager = FeaturePluginManager()
    feature_manager.find()
    for plugin_name, plugin_class in feature_manager.plugins().items():
        if plugin_name not in plugins:
            continue

        plugin_config = {"params": {}}
        for x in config["features"]:
            print(x)
            if x["type"].lower() == plugin_name.lower():
                plugin_config.update(x)
        plugin = plugin_class(config=plugin_config["params"])
        for entries_subset in entries_list:
            entries_processed = plugin(entries_subset)
            update_database(database, entries_processed)

    classifier_manager = ClassifierPluginManager()
    classifier_manager.find()
    for plugin_name, plugin_class in classifier_manager.plugins().items():
        if plugin_name not in plugins:
            continue

        plugin_config = {"params": {}}
        for x in config["classifiers"]:
            print(x)
            if x["type"].lower() == plugin_name.lower():
                plugin_config.update(x)
        plugin = plugin_class(config=plugin_config["params"])

        for entries_subset in entries_list:
            entries_processed = plugin(entries_subset)
            update_database(database, entries_processed)


def listing():
    results = []
    feature_manager = FeaturePluginManager()
    feature_manager.find()
    for plugin_name, plugin_class in feature_manager.plugins().items():
        results.append(plugin_name)

    classifier_manager = ClassifierPluginManager()
    classifier_manager.find()
    for plugin_name, plugin_class in classifier_manager.plugins().items():
        results.append(plugin_name)

    return results


def serve(config, feature_manager, classifier_manager):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    indexer_pb2_grpc.add_IndexerServicer_to_server(Commune(feature_manager, classifier_manager), server)
    port = config.get("port", 50051)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logging.info("Server is running.")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def read_config(path):
    with open(path, "r") as f:
        return json.load(f)
    return {}


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    if args.config is not None:
        config = read_config(args.config)
    else:
        config = {}

    feature_manager = FeaturePluginManager()
    feature_manager.find()

    classifier_manager = ClassifierPluginManager()
    classifier_manager.find()

    if args.mode == "local":
        plugins = listing()

        if args.plugins is not None and len(args.plugins) > 1:
            filtered_plugins = []
            for white_plugin in args.plugins:

                print(white_plugin.lower())
                for plugin in plugins:
                    if plugin.lower() == white_plugin.lower():
                        filtered_plugins.append(plugin)
            plugins = filtered_plugins
        indexing(config, args.path, args.output, args.batch)
    elif args.mode == "server":
        serve(config, feature_manager, classifier_manager)
    return 0


if __name__ == "__main__":
    sys.exit(main())
