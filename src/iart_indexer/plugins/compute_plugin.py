import importlib
import os
import re
import logging

from iart_indexer.utils.plugin.manager import Manager
from iart_indexer.utils.plugin.plugin import Plugin

from packaging import version
from typing import Any, Dict, Type
from iart_indexer.utils import convert_name


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

    def __init__(self, config: Dict = None):
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

    def __init__(self, **kwargs) -> None:
        super(ComputePlugin, self).__init__(**kwargs)

    def __call__(self, inputs, parameters: Dict = None) -> ComputePluginResult:
        return self.call(inputs, parameters)


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
