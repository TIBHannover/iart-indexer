import importlib
import os
import re
import sys
import logging

from iart_indexer.plugins.manager import PluginManager
from iart_indexer.plugins.plugin import Plugin


class FeaturePluginManager(PluginManager):
    _feature_plugins = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._feature_plugins[name] = plugin
            return plugin

        return export_helper

    def plugins(self):
        return self._feature_plugins

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "feature")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("iart_indexer.plugins.feature.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def run(self, images, plugins=None, configs=None, batchsize=128):
        print(plugins)
        plugin_list = self.init_plugins(plugins, configs)
        print(plugin_list)

        # TODO use batch size
        for image in images:
            plugin_result_list = {"id": image.id, "image": image, "plugins": []}
            for plugin in plugin_list:
                # logging.info(dir(plugin_class["plugin"]))
                plugin = plugin["plugin"]

                plugin_version = plugin.version
                plugin_name = plugin.name

                logging.info(f"Plugin start {plugin.name}:{plugin.version}")

                # exit()
                plugin_results = plugin([image])
                plugin_result_list["plugins"].append(plugin_results)

                # plugin_result_list["plugins"]plugin_results._plugin
                # # # TODO entries_processed also contains the entries zip will be

                # logging.info(f"Plugin done {plugin.name}:{plugin.version}")
                # for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):
                #     if entry.id not in plugin_result_list:
                #         plugin_result_list[entry.id] = {"image": entry, "results": []}
                #     plugin_result_list["results"].extend(annotations)
            yield plugin_result_list


class FeaturePlugin(Plugin):
    _type = "feature"

    def __init__(self, **kwargs):
        super(FeaturePlugin, self).__init__(**kwargs)

    def __call__(self, images):
        return self.call(images)


# __all__ = []
