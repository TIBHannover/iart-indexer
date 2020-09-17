from iart_indexer.plugins import ClassifierPlugin
from iart_indexer.plugins import ClassifierPluginManager
from iart_indexer.plugins import PluginResult


@ClassifierPluginManager.export("TestClassfier")
class TestClassfierPlugin(ClassifierPlugin):
    def __init__(self, **kwargs):
        super(TestClassfierPlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
