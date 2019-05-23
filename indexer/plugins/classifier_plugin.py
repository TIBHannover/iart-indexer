import os
import re
import sys

import importlib

from indexer.plugins.plugin import Plugin

_classifier_plugins = {}


def export_classifier_plugin(name):

    def export_helper(plugin):
        _classifier_plugins[name] = plugin
        return plugin

    return export_helper


def classifier_plugins():
    return _classifier_plugins


def find_plugins(path=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'classifier')):
    file_re = re.compile(r"(.+?)\.py$")
    for pl in os.listdir(path):
        match = re.match(file_re, pl)
        if match:
            a = importlib.import_module('indexer.plugins.classifier.{}'.format(match.group(1)))
            print(a)
            function_dir = dir(a)
            if "register" in function_dir:
                a.register(self)


class ClassifierPlugin(Plugin):

    _type = 'classifier'

    def __init__(self, **kwargs):
        super(ClassifierPlugin, self).__init__(**kwargs)

    def __call__(self, images):
        return self.call(images)


# __all__ = []
