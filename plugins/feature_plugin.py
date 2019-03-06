import os
import re
import sys

import importlib

_feature_plugins = {}


def export_feature_plugin(name):

    def export_helper(plugin):
        _feature_plugins[name] = plugin
        return plugin

    return export_helper


def feature_plugins():
    return _feature_plugins


def find_plugins(path=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'feature')):
    file_re = re.compile(r"(.+?)\.py$")
    for pl in os.listdir(path):
        match = re.match(file_re, pl)
        if match:
            a = importlib.import_module('plugins.feature.{}'.format(match.group(1)))
            print(a)
            function_dir = dir(a)
            if "register" in function_dir:
                a.register(self)


class FeaturePlugin():

    def __call__(self, images):
        self.call(images)


# __all__ = []
