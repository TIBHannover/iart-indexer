from iart_indexer.plugins import FeaturePlugin
from iart_indexer.plugins import FeaturePluginManager
from iart_indexer.plugins import PluginResult


@FeaturePluginManager.export("TestFeature")
class TestFeaturePlugin(FeaturePlugin):
    def __init__(self, **kwargs):
        super(TestFeaturePlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
