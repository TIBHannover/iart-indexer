import logging
import requests
from typing import Any, Dict
from ray import serve
from ray.serve import Application
from iart_indexer.plugins.compute_plugin import ComputePlugin, ComputePluginManager
from iart_indexer.inference import InferenceServer, InferenceServerFactory
from google.protobuf.json_format import MessageToDict, ParseDict, Parse


@serve.deployment
class Deployment:
    def __init__(self, plugin: ComputePlugin) -> None:
        self.plugin = plugin

    async def __call__(self, request) -> Dict[str, str]:
        data = await request.json()

        from iart_indexer.indexer_pb2 import AnalyseRequest

        analyse_request = ParseDict(data["inputs"], AnalyseRequest())

        results = self.plugin(analyse_request)

        return MessageToDict(results)


def app_builder(args) -> Application:
    logging.warning(args)
    return Deployment.options(**args["options"]).bind(args["plugin"])


@InferenceServerFactory.export("ray")
class RayInferenceServer(InferenceServer):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)

    def __call__(self, plugin: str, inputs: Dict, parameters: Dict) -> Dict:
        results = requests.post(
            f"http://inference_ray:8000{plugin_to_run['route']}",
            json={
                "inputs": inputs,
                "parameters": parameters,
            },
        ).json()
        return results

    def start(self, compute_plugin_manager: ComputePluginManager) -> None:
        for compute_plugin in compute_plugin_manager.plugin_list:
            plugin_cls = compute_plugin["plugin_cls"]
            plugin_config = compute_plugin["config"]

            if "inference" in plugin_config:
                inference_config = plugin_config["inference"]
            else:
                inference_config = dict()

            if "requirements" in inference_config:
                requirements = inference_config["requirements"]
            else:
                requirements = list()
            serve.run(
                app_builder(
                    {
                        "plugin": plugin_cls(
                            config=plugin_config, compute_plugin_manager=compute_plugin_manager, inference_server=self
                        ),
                        "options": {
                            "name": plugin_cls.name,
                            "ray_actor_options": {"runtime_env": {"pip": requirements}},
                            "autoscaling_config": {"min_replicas": 1},
                        },
                    }
                ),
                route_prefix=f"/{plugin_cls.name}",
                name=plugin_cls.name,
            )

    def __call__(self, compute_plugin_manager, plugin, request):
        from iart_indexer import indexer_pb2

        found = False
        for compute_plugin in compute_plugin_manager.plugin_list:
            plugin_key = compute_plugin["plugin_key"]

            plugin_cls = compute_plugin["plugin_cls"]
            plugin_config = compute_plugin["config"]

            if plugin == plugin_key:
                found = True
                break

        if not found:
            logging.error(f"{plugin} not found")
            return None

        results = ParseDict(
            requests.post(
                f"http://localhost:8000/{plugin_cls.name}",
                json={
                    "inputs": MessageToDict(request),
                },
            ).json(),
            indexer_pb2.AnalyseReply(),
        )
        # logging.info(f"[AnalyserPluginManager] {run_id} plugin: {plugin_to_run}")
        # logging.info(f"[AnalyserPluginManager] {run_id} data: {[{k:x.id} for k,x in inputs.items()]}")
        # logging.info(f"[AnalyserPluginManager] {run_id} parameters: {parameters}")
        # results = plugin_to_run(inputs, data_manager, parameters, callbacks)
        # logging.info(f"[AnalyserPluginManager] {run_id} results: {[{k:x.id} for k,x in results.items()]}")

        return results
