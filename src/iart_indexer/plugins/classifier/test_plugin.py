from iart_indexer.plugins import ClassifierPlugin, ClassifierPluginManager, PluginResult


@ClassifierPluginManager.export("TestClassfier")
class TestClassfierPlugin(ClassifierPlugin):
    def __init__(self, **kwargs):
        super(TestClassfierPlugin, self).__init__(**kwargs)

    def call(self, entries):
        return PluginResult(self, [], [])
