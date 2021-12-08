import os
import re
import sys
import logging
import importlib

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
                function_dir = dir(a)

                if "register" in function_dir:
                    a.register(self)

    def indexing(
        self,
        train_entries,
        index_entries,
        collections=None,
        rebuild=False,
        plugins=None,
        configs=None,
    ):
        # TODO add lock here
        logging.info("IndexerPluginManager: indexing")
        logging.info(f"IndexerPluginManager: {len(self.plugin_list)} {collections}")

        for plugin in self.plugin_list:
            plugin = plugin["plugin"]
            logging.info(f"IndexerPluginManager: {plugin.name}")

            plugin.indexing(
                train_entries,
                index_entries,
                rebuild=rebuild,
                collections=collections,
            )

        # TODO force reloading of other processes

    def search(
        self,
        queries,
        collections=None,
        include_default_collection=True,
        size=100,
    ):
        result_list = []

        for plugin in self.plugin_list:
            plugin = plugin["plugin"]

            entries = plugin.search(
                queries,
                collections=collections, 
                include_default_collection=include_default_collection,
                size=size
            )

            result_list.extend(entries)

        return result_list


class IndexerPlugin(Plugin):
    _type = "indexer"

    def __init__(self, **kwargs):
        super(IndexerPlugin, self).__init__(**kwargs)

    def indexing(self, train_entries, index_entries):
        pass

    def search(self, queries, size=100):
        pass
