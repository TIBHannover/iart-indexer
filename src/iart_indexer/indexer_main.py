import argparse
import json
import logging
import os
import re
import sys
import uuid

import imageio

from iart_indexer.client import Client
from iart_indexer.server import Server


def parse_args():
    parser = argparse.ArgumentParser(description="Indexing a set of images")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    # parser.add_argument('-l', '--list', help='list all plugins')

    parser.add_argument("--host", help="")
    parser.add_argument("--port", type=int, help="")
    parser.add_argument("--plugins", nargs="+", help="")
    parser.add_argument("--image_input", help="")
    parser.add_argument("--batch", default=512, type=int, help="split images in batch")

    parser.add_argument(
        "--task", choices=["list_plugins", "copy_images", "indexing", "build_suggester"], help="verbose output"
    )

    parser.add_argument("--path", help="path to image or folder to indexing")

    parser.add_argument("--output", help="copy image to new folder with hash id")
    parser.add_argument("--image_output", help="copy image to new folder with hash id")

    parser.add_argument("-c", "--config", help="config path")
    parser.add_argument("-m", "--mode", choices=["client", "server"], default="local", help="verbose output")
    args = parser.parse_args()
    return args


def read_config(path):
    with open(path, "r") as f:
        return json.load(f)
    return {}


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    if args.config is not None:
        config = read_config(args.config)
    else:
        config = {}

    if args.mode == "client":
        if args.host is not None:
            config["host"] = args.host

        if args.port is not None:
            config["port"] = args.port

        client = Client(config)
        available_plugins = client.plugin_list()
        if args.task == "list_plugins":
            print(available_plugins)
        elif args.task == "copy_images":
            entries = client.copy_images(paths=args.path, image_input=args.image_input, image_output=args.image_output)
            if args.output is not None:
                with open(args.output, "w") as f:
                    for line in entries:
                        f.write(json.dumps(line) + "\n")
        elif args.task == "indexing":
            plugins = []
            plugins_selected = None
            if args.plugins:
                plugins_selected = [x.lower() for x in args.plugins]
            for t, plugin_list in available_plugins.items():
                for plugin in plugin_list:
                    if plugins_selected is not None:
                        if plugin.lower() in plugins_selected:
                            plugins.append(plugin)
                    else:
                        plugins.append(plugin)

            client.indexing(paths=args.path, image_input=args.image_input, plugins=plugins)

        elif args.task == "build_suggester":

            client.build_suggester()

    elif args.mode == "server":
        server = Server(config)
        server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
