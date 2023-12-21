from typing import Dict
import logging

from iart_indexer.utils.plugin import Plugin
from iart_indexer.utils.plugin import Factory


class InferenceServer(Plugin):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)

    def start(self) -> None:
        pass


class InferenceServerFactory(Factory):
    _plugins = {}

    @classmethod
    def export(cls, name: str) -> None:
        def export_helper(plugin):
            cls._plugins[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def build(cls, name: str, config: Dict = None) -> InferenceServer:
        if name not in cls._plugins:
            logging.error(f"Unknown cache server: {name}")
            return None

        return cls._plugins[name](config)

    @classmethod
    def __call__(cls, name: str, config: Dict = None) -> InferenceServer:
        if name not in cls._plugins:
            logging.error(f"Unknown cache server: {name}")
            return None

        return cls._plugins[name](config)
