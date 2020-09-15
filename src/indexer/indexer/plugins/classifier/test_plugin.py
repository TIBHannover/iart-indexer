from indexer.indexer.plugins import ClassifierPlugin
from indexer.indexer.plugins import ClassifierPluginManager
from indexer.indexer.plugins import PluginResult


@ClassifierPluginManager.export("TestClassfier")
class TestClassfierPlugin(ClassifierPlugin):
    def __init__(self, **kwargs):
        super(TestClassfierPlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
