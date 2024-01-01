import argparse
import json
import logging
import os
import re
import sys
import uuid
import time

import imageio

from iart_indexer.client import Client
from iart_indexer.server import Server


def parse_args():
    parser = argparse.ArgumentParser(description="Indexing a set of images")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="verbose output")
    # parser.add_argument('-l', '--list', help='list all plugins')

    parser.add_argument("--host", help="")
    parser.add_argument("--port", type=int, help="")
    parser.add_argument("--plugins", nargs="+", help="")
    parser.add_argument("--image_paths", help="")
    parser.add_argument("--analyse_inputs", help="")
    parser.add_argument("--analyse_parameters", help="")
    parser.add_argument("--query", help="")
    parser.add_argument("--batch", default=512, type=int, help="split images in batch")

    parser.add_argument(
        "--task",
        choices=[
            "list_plugins",
            "copy_images",
            "indexing",
            "bulk_indexing",
            "build_suggester",
            "get",
            "analyse",
            "search",
            "aggregate",
            "build_feature_cache",
            "build_indexer",
            "load",
            "dump",
        ],
        help="verbose output",
    )

    parser.add_argument("--dump_path", help="path to image or folder to indexing")
    parser.add_argument("--dump_origin", help="name of a collection to dump")
    parser.add_argument("--path", help="path to image or folder to indexing")
    parser.add_argument("--id", help="id for entry query")
    parser.add_argument("--field_name", nargs="+", help="id for entry query")

    parser.add_argument("--output", help="copy image to new folder with hash id")
    parser.add_argument("--image_output", help="copy image to new folder with hash id")

    parser.add_argument("--aggr_part", help="id for entry query")
    parser.add_argument("--aggr_type", help="id for entry query")
    parser.add_argument("--aggr_field_name", help="id for entry query")
    parser.add_argument("--aggr_size", type=int, help="id for entry query")

    parser.add_argument("--rebuild", action="store_true", help="verbose output")
    parser.add_argument("--collections", nargs="+", help="id for entry query")

    parser.add_argument("-c", "--config", help="config path")
    parser.add_argument("-m", "--mode", choices=["client", "server"], default="client", help="verbose output")
    args = parser.parse_args()
    return args


def read_config(path):
    with open(path, "r") as f:
        return json.load(f)
    return {}


def main():
    args = parse_args()
    level = logging.ERROR
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

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
        if args.task == "list_plugins":
            available_plugins = client.plugin_list()
            print(available_plugins)
        elif args.task == "copy_images":
            entries = client.copy_images(paths=args.path, image_paths=args.image_paths, image_output=args.image_output)
            if args.output is not None:
                with open(args.output, "w") as f:
                    for line in entries:
                        f.write(json.dumps(line) + "\n")
        elif args.task == "indexing":
            available_plugins = client.plugin_list()
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

            client.indexing(paths=args.path, image_paths=args.image_paths, plugins=plugins)

        elif args.task == "bulk_indexing":
            available_plugins = client.plugin_list()
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

            client.bulk_indexing(paths=args.path, image_paths=args.image_paths, plugins=plugins)

        elif args.task == "build_suggester":
            client.build_suggester(args.field_name)

        elif args.task == "get":
            print(client.get(args.id))

        elif args.task == "search":
            # try:
            query = json.loads(args.query)
            # except:
            #     query = {"queries": [{"type": "meta", "query": args.query}]}
            time_start = time.time()
            client.search(query)
            time_stop = time.time()
            print(time_stop - time_start)

        elif args.task == "build_indexer":
            client.build_indexer(rebuild=args.rebuild, collections=args.collections)

        elif args.task == "build_feature_cache":
            client.build_feature_cache()

        elif args.task == "build_feature_cache":
            client.build_feature_cache()

        elif args.task == "dump":
            client.dump(args.dump_path, args.dump_origin)

        elif args.task == "load":
            client.load(args.dump_path)

        elif args.task == "aggregate":
            print(
                client.aggregate(
                    part=args.aggr_part, type=args.aggr_type, field_name=args.aggr_field_name, size=args.aggr_size
                )
            )

        elif args.task == "analyse":
            available_plugins = client.plugin_list()
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
            print(client.analyse(json.loads(args.analyse_inputs), json.loads(args.analyse_parameters), plugins[0]))

    elif args.mode == "server":
        server = Server(config)
        server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
