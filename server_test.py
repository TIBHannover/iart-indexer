import indexer_pb2
import indexer_pb2_grpc

import grpc
import time


def get_plugin_list():
    result = {"response": []}
    channel = grpc.insecure_channel("localhost:50051")
    stub = indexer_pb2_grpc.IndexerStub(channel)
    response = stub.list_plugins(indexer_pb2.ListPluginsRequest())

    for plugin in response.plugins:
        tmp = {"plugin": plugin.name, "type": plugin.type, "settings": []}

        for setting in plugin.settings:
            tmp["settings"].append({"name": setting.name, "default": setting.default, "type": setting.type})

        result["response"].append(tmp)

    return result


print(get_plugin_list())


def run_plugin(plugin, image):

    result = {"response": []}
    channel = grpc.insecure_channel("localhost:50051")
    stub = indexer_pb2_grpc.IndexerStub(channel)
    request = indexer_pb2.IndexingRequest()
    request.update_database = False
    request_plugin = request.plugins.add()
    request_plugin.name = "YUVHistogramFeature"
    request_image = request.images.add()
    request_image.path = image.encode()
    response = stub.indexing(request)

    return response.id


def status(job_id):

    channel = grpc.insecure_channel("localhost:50051")
    stub = indexer_pb2_grpc.IndexerStub(channel)
    request = indexer_pb2.StatusRequest()
    request.id = job_id
    response = stub.status(request)

    return response


job_id = run_plugin("yuv_histogram", image="/home/matthias/images/test_2.jpg")
print(job_id)

print(status(job_id))
time.sleep(1)
print(status(job_id))
