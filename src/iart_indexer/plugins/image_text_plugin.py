import importlib
import os
import re
import sys
import logging

# from typing import Union

from iart_indexer.plugins.manager import PluginManager
from iart_indexer.plugins.plugin import Plugin

from packaging import version


class ImageTextPluginManager(PluginManager):
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

    def find(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "image_text")):
        file_re = re.compile(r"(.+?)\.py$")
        for pl in os.listdir(path):
            match = re.match(file_re, pl)
            if match:
                a = importlib.import_module("iart_indexer.plugins.image_text.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)

    def run(self, text, filter_plugins=None, plugins=None, configs=None, batchsize=128):

        plugin_list = self.init_plugins(plugins, configs)
        logging.info('#########################')
        logging.info(plugin_list)

        # print(f"PLUGINS: {plugin_list}")
        if filter_plugins is None:
            filter_plugins = [[]] * len(text)
        # TODO use batch size
        # print(f"LEN1 {len(images)} ")
        # print(f"LEN2 {len(filter_plugins)} ")
        # print(f"{filter_plugins}")
        for (image, filters) in zip(text, filter_plugins):
            # print("IMAGE")
            plugin_result_list = {"text": text, "plugins": []}
            for plugin in plugin_list:
                # print(f"PLUGIN: {plugin}")
                # logging.info(dir(plugin_class["plugin"]))
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


class ImageTextPlugin(Plugin):
    _type = "image_text"

    def __init__(self, **kwargs):
        super(ImageTextPlugin, self).__init__(**kwargs)

    def __call__(self, images):
        return self.call(images)


# __all__ = []
