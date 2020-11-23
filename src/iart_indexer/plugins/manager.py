class PluginManager:
    def __init__(self, configs=None):
        self.configs = configs
        if configs is None:
            self.configs = []

    def plugins(self):
        return {}

    def init_plugins(self, plugins=None, configs=None):
        if plugins is None:
            plugins = list(self.plugins().keys())

        # TODO add merge tools
        if configs is None:
            configs = self.configs

        plugin_list = []
        plugin_name_list = [x.lower() for x in plugins]
        print(f"MANAGER: {plugin_name_list}")

        for plugin_name, plugin_class in self.plugins().items():
            print(f"MANAGER: {plugin_name.lower()}")
            if plugin_name.lower() not in plugin_name_list:
                continue
            print(f"MANAGER: found")

            plugin_config = {"params": {}}
            for x in self.configs:
                if x["type"].lower() == plugin_name.lower():
                    plugin_config.update(x)

            plugin = plugin_class(config=plugin_config["params"])
            plugin_list.append({"plugin": plugin, "plugin_cls": plugin_class, "config": plugin_config})
            print(f"MANAGER: {plugin_list}")

        return plugin_list
