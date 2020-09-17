from iart_indexer.plugins import FeaturePlugin, FeaturePluginManager, PluginResult


@FeaturePluginManager.export("TestFeature")
class TestFeaturePlugin(FeaturePlugin):
    def __init__(self, **kwargs):
        super(TestFeaturePlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
