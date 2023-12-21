from iart_indexer.plugins import ComputePlugin, ComputePluginManager, ComputePluginResult


@ComputePluginManager.export("TestFeature")
class TestFeaturePlugin(ComputePlugin):
    def __init__(self, **kwargs):
        super(TestFeaturePlugin, self).__init__(**kwargs)

    def call(self, entries):
        return ComputePluginResult(self, [], [])
