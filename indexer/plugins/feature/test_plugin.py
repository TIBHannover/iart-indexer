from indexer.plugins import FeaturePlugin
from indexer.plugins import FeaturePluginManager
from indexer.plugins import PluginResult


@FeaturePluginManager.export('TestFeature')
class TestFeaturePlugin(FeaturePlugin):

    def __init__(self, **kwargs):
        super(TestFeaturePlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
