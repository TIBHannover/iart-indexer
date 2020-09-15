import os
import re
import sys

import importlib

from indexer.indexer.plugins.plugin import Plugin


class FeaturePluginManager:
    _feature_plugins = {}

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
                a = importlib.import_module("indexer.indexer.plugins.feature.{}".format(match.group(1)))
                # print(a)
                function_dir = dir(a)
                if "register" in function_dir:
                    a.register(self)


class FeaturePlugin(Plugin):
    _type = "feature"

    def __init__(self, **kwargs):
        super(FeaturePlugin, self).__init__(**kwargs)

    def __call__(self, images):
        return self.call(images)


# __all__ = []
