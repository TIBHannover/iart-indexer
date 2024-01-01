import importlib
import os
import re
import logging

from iart_indexer.utils.plugin.manager import Manager
from iart_indexer.utils.plugin.plugin import Plugin

from packaging import version
from typing import Any, Dict, Type
from iart_indexer.utils import convert_name

from iart_indexer import indexer_pb2


class ComputePluginResult:
    def __init__(self, plugin, entries, annotations):
        self._plugin = plugin
        self._entries = entries
        self._annotations = annotations
        assert len(self._entries) == len(self._annotations)

    def __repr__(self):
        return f"{self._plugin} {self._annotations}"


class ComputePlugin(Plugin):
    @classmethod
    def __init_subclass__(
        cls,
        parameters: Dict[str, Any] = None,
        # requires: Dict[str, Type[Data]] = None,
        # provides: Dict[str, Type[Data]] = None,
        requires: Dict[str, Type] = None,
        provides: Dict[str, Type] = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._requires = requires
        cls._provides = provides
        cls._parameters = parameters
        cls._name = convert_name(cls.__name__)

    def __init__(
        self,
        compute_plugin_manager: "ComputePluginManager",
        inference_server: "InferenceServer",
        config: Dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._compute_plugin_manager = compute_plugin_manager
        self._inference_server = inference_server
        self._config = self._default_config
        if config is not None:
            self._config.update(config)

    @classmethod
    @property
    def requires(cls):
        return cls._requires

    @classmethod
    @property
    def provides(cls):
        return cls._provides

    @property
    def compute_plugin_manager(self):
        return self._compute_plugin_manager

    @property
    def inference_server(self):
        return self._inference_server

    @staticmethod
    def map_analyser_request_to_dict(request):
        input_dict = {}
        for input in request.inputs:
            if input.name not in input_dict:
                input_dict[input.name] = []
            # logging.error(input)
            if input.WhichOneof("data") == "image":
                input_dict[input.name].append({"type": "image", "content": input.image.content})
            if input.WhichOneof("data") == "string":
                input_dict[input.name].append({"type": "string", "content": input.string.text})

        parameter_dict = {}
        for parameter in request.parameters:
            if parameter.name not in parameter_dict:
                parameter_dict[parameter.name] = []
            # TODO convert datatype
            if parameter.type == indexer_pb2.FLOAT_TYPE:
                parameter_dict[parameter.name] = float(parameter.content)
            if parameter.type == indexer_pb2.INT_TYPE:
                parameter_dict[parameter.name] = int(parameter.content)
            if parameter.type == indexer_pb2.STRING_TYPE:
                parameter_dict[parameter.name] = str(parameter.content)
            if parameter.type == indexer_pb2.BOOL_TYPE:
                parameter_dict[parameter.name] = bool(parameter.content)

        return input_dict, parameter_dict

    @staticmethod
    def map_dict_to_analyser_request(inputs, parameters):
        request = indexer_pb2.AnalyseRequest()
        for key, values in inputs.items():
            for value in values:
                input_field = request.inputs.add()
                if value["type"] == "image":
                    input_field.name = "image"
                    if "path" in value:
                        input_field.image.content = open(value["path"], "rb").read()
                    elif "content" in value:
                        input_field.image.content = value["content"]
                    else:
                        logging.error("Missing image content")

                elif value["type"] == "string":
                    input_field.name = "text"
                    input_field.string.text = value["content"]

        for key, value in parameters.items():
            parameter = request.parameters.add()
            parameter.content = str(value)
            parameter.name = key

            if isinstance(value, float):
                parameter.type = indexer_pb2.FLOAT_TYPE
            if isinstance(value, int):
                parameter.type = indexer_pb2.INT_TYPE
            if isinstance(value, str):
                parameter.type = indexer_pb2.STRING_TYPE

        return request

    def __call__(self, analyse_request: indexer_pb2.AnalyseRequest) -> ComputePluginResult:
        return self.call(analyse_request)


# __all__ = []


class ComputePluginManager(Manager):
    _plugins = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.find()
        self.plugin_list = self.build_plugin_list()

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._plugins[name] = plugin
            return plugin

        return export_helper

    def plugins(self):
        return self._plugins

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "compute")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("iart_indexer.plugins.compute.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def build_plugin(self, plugin: str, config: Dict = None) -> ComputePlugin:
        if plugin not in self._plugins:
            return None
        plugin_to_run = None
        for plugin_name, plugin_cls in self._plugins.items():
            if plugin_name == plugin:
                plugin_to_run = plugin_cls

        if plugin_to_run is None:
            logging.error(f"[AnalyserPluginManager] plugin: {plugin} not found")
            return None

        return plugin_to_run(config)

    def run(self, images, filter_plugins=None, plugins=None, configs=None, batchsize=128):
        if plugins is None and configs is None:
            plugins = self.plugin_list

        if filter_plugins is None:
            filter_plugins = [] * len(images)
        # TODO use batch size
        for image, filters in zip(images, filter_plugins):
            plugin_result_list = {"image": image, "plugins": []}
            for plugin in plugins:
                plugin = plugin["plugin"]
                plugin_version = version.parse(str(plugin.version))

                founded = False
                for f in filters:
                    f_version = version.parse(str(f["version"]))
                    if f["plugin"] == plugin.name and f_version >= plugin_version:
                        founded = True

                if founded:
                    continue

                logging.info(f"Plugin start {plugin.name}:{plugin.version}")

                plugin_results = plugin([image])
                plugin_result_list["plugins"].append(plugin_results)

            yield plugin_result_list
