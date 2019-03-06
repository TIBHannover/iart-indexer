from plugins import FeaturePlugin
from plugins import export_feature_plugin


@export_feature_plugin('TestFeature')
class TestFeaturePlugin(FeaturePlugin):

    def __init__(self):
        pass

    def call(self, image):
        print('TestPlugin')
        return 0
