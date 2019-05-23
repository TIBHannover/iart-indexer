from indexer.plugins import ClassifierPlugin
from indexer.plugins import export_classifier_plugin
from indexer.plugins import PluginResult


@export_classifier_plugin('TestClassfier')
class TestClassfierPlugin(ClassifierPlugin):

    def __init__(self, **kwargs):
        super(TestClassfierPlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
