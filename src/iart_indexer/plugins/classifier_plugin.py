import importlib
import os
import re
import sys
import logging

from iart_indexer.plugins.manager import PluginManager
from iart_indexer.plugins.plugin import Plugin

from packaging import version


class ClassifierPluginManager(PluginManager):
    _classifier_plugins = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.find()
        self.plugin_list = self.init_plugins()

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._classifier_plugins[name] = plugin
            return plugin

        return export_helper

    def plugins(self):
        return self._classifier_plugins

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "classifier")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("iart_indexer.plugins.classifier.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def run(self, images, filter_plugins=None, plugins=None, configs=None, batchsize=128):
        if plugins is None and configs is None:
            plugin_list = self.plugin_list

        if filter_plugins is None:
            filter_plugins = [] * len(images)
        # TODO use batch size
        for (image, filters) in zip(images, filter_plugins):
            plugin_result_list = {"image": image, "plugins": []}
            for plugin in plugin_list:

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


class ClassifierPlugin(Plugin):

    _type = "classifier"

    def __init__(self, **kwargs):
        super(ClassifierPlugin, self).__init__(**kwargs)

    def __call__(self, images):
        return self.call(images)


# __all__ = []
