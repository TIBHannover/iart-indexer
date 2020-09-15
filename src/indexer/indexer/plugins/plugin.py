from indexer.indexer.utils import convert_name


class PluginResult:
    def __init__(self, plugin, entries, annotations):
        self._plugin = plugin
        self._entries = entries
        self._annotations = annotations
        assert len(self._entries) == len(self._annotations)

    def __repr__(self):
        return f"{self._plugin} {self._annotations}"


class Plugin:

    default_config = {}

    default_version = 0.1

    _type = ""

    def __init__(self, config=None, name=None):
        self._config = self.default_config
        if config is not None:
            self._config.update(config)

        self._version = self.default_version

        if name is None:
            name = convert_name(self.__class__.__name__)

        self._name = name

    @property
    def config(self):
        return self._config

    @property
    def version(self):
        return self._version

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
