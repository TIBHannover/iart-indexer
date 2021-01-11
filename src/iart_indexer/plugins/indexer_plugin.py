import importlib
import os
import re
import sys
import logging

from iart_indexer.plugins.manager import PluginManager
from iart_indexer.plugins.plugin import Plugin


class IndexerPluginManager(PluginManager):
    _indexer_plugins = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.find()
        self.plugin_list = self.init_plugins()

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._indexer_plugins[name] = plugin
            return plugin

        return export_helper

    def plugins(self):
        return self._indexer_plugins

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "indexer")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("iart_indexer.plugins.indexer.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def indexing(self, entries, plugins=None, configs=None):
        print("IndexerPluginManager: START")
        plugin_list = self.init_plugins(plugins, configs)
        print(f"IndexerPluginManager: {len(plugin_list)}")
        for plugin in plugin_list:
            plugin = plugin["plugin"]
            print(f"IndexerPluginManager: {plugin.name}")
            plugin.indexing(entries)
            # print(x)

    def search(self, queries, size=100):
        result_list = []
        for plugin in self.plugin_list:
            plugin = plugin["plugin"]
            entries = plugin.search(queries, size=size)
            result_list.extend(entries)

        return result_list

    # def indexing(self, entries, plugins=None, configs=None):
    #     print("############")
    #     print(plugins)
    #     plugin_list = self.init_plugins(plugins, configs)
    #     print(plugin_list)

    #     # TODO use batch size
    #     for image in entries:
    #         plugin_result_list = {"id": image.id, "image": image, "plugins": []}
    #         for plugin in plugin_list:
    #             # logging.info(dir(plugin_class["plugin"]))
    #             plugin = plugin["plugin"]

    #             plugin_version = plugin.version
    #             plugin_name = plugin.name

    #             logging.info(f"Plugin start {plugin.name}:{plugin.version}")

    #             plugin_results = plugin([image])
    #             plugin_result_list["plugins"].append(plugin_results)

    #             # plugin_result_list["plugins"]plugin_results._plugin
    #             # # # TODO entries_processed also contains the entries zip will be

    #             # logging.info(f"Plugin done {plugin.name}:{plugin.version}")
    #             # for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):
    #             #     if entry.id not in plugin_result_list:
    #             #         plugin_result_list[entry.id] = {"image": entry, "results": []}
    #             #     plugin_result_list["results"].extend(annotations)
    #         yield plugin_result_list


class IndexerPlugin(Plugin):
    _type = "indexer"

    def __init__(self, **kwargs):
        super(IndexerPlugin, self).__init__(**kwargs)

    def indexing(self, images):
        pass

    def search(self, queries, size=100):
        pass
