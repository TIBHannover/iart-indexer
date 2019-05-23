from indexer.plugins import FeaturePlugin
from indexer.plugins import export_feature_plugin
from indexer.plugins import PluginResult


@export_feature_plugin('TestFeature')
class TestFeaturePlugin(FeaturePlugin):

    def __init__(self, **kwargs):
        super(TestFeaturePlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
