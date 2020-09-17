import logging
import threading
import time
import uuid
from concurrent import futures

import grpc

from iart_indexer import indexer_pb2, indexer_pb2_grpc
from iart_indexer.database.elasticsearch_database import ElasticSearchDatabase
from iart_indexer.plugins import *
from iart_indexer.plugins import ClassifierPlugin, FeaturePlugin
from iart_indexer.utils import image_from_proto, meta_from_proto

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def compute_plugins(args):
    logging.info("Start computing job")
    plugin_classes = args["plugin_classes"]
    images = args["images"]
    database = args["database"]
    for x in images:
        exist_entry = database.get_entry(x.id)
        if exist_entry is None:
            meta = meta_from_proto(x.meta)
            database.insert_entry(
                x.id,
                {"id": x.id, "meta": meta},
            )

    # if database is not None:
    #     existing_hash = [x["id"] for x in list(database.all())]
    #     for x in images:
    #         if x.id not in existing_hash:

    plugin_result_list = {}
    for plugin_class in plugin_classes:
        print(f"Plugin start {plugin_class}")
        plugin = plugin_class["plugin"](config=plugin_class["config"]["params"])
        plugin_results = plugin(images)
        # # TODO entries_processed also contains the entries zip will be

        print(f"Plugin done {plugin_class}")
        for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):
            if entry.id not in plugin_result_list:
                plugin_result_list[entry.id] = {"image": entry, "results": []}
            plugin_result_list[entry.id]["results"].extend(annotations)
        if database is not None:
            update_database(database, plugin_results)
            print(f"Plugin result save {plugin_class}")

    return indexer_pb2.IndexingResult(
        results=[indexer_pb2.ImageResult(image=x["image"], results=x["results"]) for x in plugin_result_list.values()]
    )


class Commune(indexer_pb2_grpc.IndexerServicer):
    def __init__(self, config, feature_manager, classifier_manager):
        self.config = config
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

            plugin_config = {"params": {}}
            for x in self.config["features"]:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)
            plugin_list.append({"plugin": plugin_class, "config": plugin_config})

        for plugin_name, plugin_class in self.classifier_manager.plugins().items():
            if plugin_name.lower() not in plugin_name_list:
                continue
            plugin_config = {"params": {}}
            for x in self.config["classifier"]:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)
            plugin_list.append({"plugin": plugin_class, "config": plugin_config})
        database = None
        if request.update_database:
            database = ElasticSearchDatabase(config=self.config.get("elasticsearch", {}))

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
            return indexer_pb2.StatusReply(status="done", result=result)

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

        hash_id = entry.id

        database.update_plugin(
            hash_id,
            plugin_name=plugin_results._plugin.name,
            plugin_version=plugin_results._plugin.version,
            plugin_type=plugin_results._plugin.type,
            annotations=annotations,
        )


class Server:
    def __init__(self, config):
        self.config = config

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        self.feature_manager = FeaturePluginManager()
        self.feature_manager.find()

        self.classifier_manager = ClassifierPluginManager()
        self.classifier_manager.find()

        indexer_pb2_grpc.add_IndexerServicer_to_server(
            Commune(config, self.feature_manager, self.classifier_manager), self.server
        )
        grpc_config = config.get("grpc", {})
        port = grpc_config.get("port", 50051)

        self.server.add_insecure_port(f"[::]:{port}")

    def run(self):
        self.server.start()
        logging.info("Server is now running.")
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            self.server.stop(0)
