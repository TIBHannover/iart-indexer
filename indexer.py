import os
import sys
import re
import json
import argparse
import logging
import uuid
import imageio

import tensorflow as tf

from indexer.plugins import *
from indexer.config import IndexerConfig

from database.elasticsearch_database import ElasticSearchDatabase

from indexer.utils import copy_image_hash, filename_without_ext, image_resolution


def parse_args():
    parser = argparse.ArgumentParser(description='Indexing a set of images')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-l', '--list', action='store_true', help='list all plugins')
    parser.add_argument('--plugins', nargs='+', default=[], help='list all plugins')
    parser.add_argument('-p', '--path', help='path to image or folder to indexing')

    parser.add_argument('-b', '--batch', default=512, type=int, help='split images in batch')
    parser.add_argument('-o', '--output', help='copy image to new folder with hash id')
    # parser.add_argument('-u', '--update', action='store_true', help='only reindexing existing documents')
    parser.add_argument('-j', '--jsonl', help='add all found documents to the index')
    # parser.add_argument('-d', '--database', help='database type for index')
    parser.add_argument('-c', '--config', help='config path')
    parser.add_argument('-m', '--mode', choices=['local', 'server'], default='local', help='verbose output')
    args = parser.parse_args()
    return args


def update_database(database, plugin_results):
    for entry, annotations in zip(plugin_results._entries, plugin_results._annotations):

        hash_id = entry['id']

        database.update_plugin(hash_id,
                               plugin_name=plugin_results._plugin.name,
                               plugin_version=plugin_results._plugin.version,
                               plugin_type=plugin_results._plugin.type,
                               annotations=annotations)


def list_images(paths, name_as_hash=False):
    if not isinstance(paths, (list, set)):

        if os.path.isdir(paths):
            file_paths = []
            for root, dirs, files in os.walk(paths):
                for f in files:
                    file_path = os.path.abspath(os.path.join(root, f))
                    # TODO filter
                    file_paths.append(file_path)

            paths = file_paths
        else:
            paths = [os.path.abspath(paths)]

    entries = [{
        'id': os.path.splitext(os.path.basename(path))[0] if name_as_hash else uuid.uuid4().hex,
        'filename': os.path.basename(path),
        'path': os.path.abspath(path),
        'meta': []
    } for path in paths]

    return entries


def list_jsonl(paths):
    entries = []
    with open(paths, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

    return entries


def copy_images(entries, output, resolutions=[500, 200, -1]):
    entires_result = []
    for i in range(len(entries)):

        entry = entries[i]
        print(entry['path'])
        copy_result = copy_image_hash(entry['path'], output, entry['id'], resolutions)
        if copy_result is not None:
            hash_value, path = copy_result
            entry.update({'path': path})
            entires_result.append(entry)

    return entires_result


def split_batch(entries, batch_size=512):
    if batch_size < 1:
        return [entries]

    return [entries[x * batch_size:(x + 1) * batch_size] for x in range(len(entries) // batch_size + 1)]


def indexing(paths, output, batch_size: int = 512, plugins: list = [], config: dict = {}):

    # TODO replace with abstract class
    if 'db' in config:
        database = ElasticSearchDatabase(config=config['db'])

    # handel images or jsonl
    if paths is not None:
        if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == '.jsonl':
            entries = list_jsonl(paths)
        else:
            entries = list_images(paths)

        if output:
            entries = copy_images(entries, output)

    else:
        entries = list(database.all())

    # TODO scroll
    existing_hash = [x['id'] for x in list(database.all())]
    for x in entries:
        if x['id'] not in existing_hash:
            resolution = image_resolution(x['path'])
            if resolution is None:
                continue
            database.insert_entry(
                x['id'], {
                    'id': x['id'],
                    'path': x['path'],
                    'filename': x['filename'],
                    'meta': x['meta'],
                    'image': {
                        'height': resolution[0],
                        'width': resolution[1]
                    }
                })

    logging.info(f'Indexing {len(entries)} documents')

    entries_list = split_batch(entries, batch_size)

    feature_manager = FeaturePluginManager()
    feature_manager.find()
    for plugin_name, plugin_class in feature_manager.plugins().items():
        if plugin_name not in plugins:
            continue

        plugin_config = {'params': {}}
        for x in config['features']:
            print(x)
            if x['type'].lower() == plugin_name.lower():
                plugin_config.update(x)
        plugin = plugin_class(config=plugin_config['params'])
        for entries_subset in entries_list:
            entries_processed = plugin(entries_subset)
            update_database(database, entries_processed)

    classifier_manager = ClassifierPluginManager()
    classifier_manager.find()
    for plugin_name, plugin_class in classifier_manager.plugins().items():
        if plugin_name not in plugins:
            continue

        plugin_config = {'params': {}}
        for x in config['classifiers']:
            print(x)
            if x['type'].lower() == plugin_name.lower():
                plugin_config.update(x)
        plugin = plugin_class(config=plugin_config['params'])

        for entries_subset in entries_list:
            entries_processed = plugin(entries_subset)
            update_database(database, entries_processed)


def listing():
    results = []
    feature_manager = FeaturePluginManager()
    feature_manager.find()
    for plugin_name, plugin_class in feature_manager.plugins().items():
        results.append(plugin_name)

    classifier_manager = ClassifierPluginManager()
    classifier_manager.find()
    for plugin_name, plugin_class in classifier_manager.plugins().items():
        results.append(plugin_name)

    return results


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    plugins = listing()

    if args.list:
        for x in plugins:
            print(x)
        return

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    if args.plugins is not None and len(args.plugins) > 1:
        filtered_plugins = []
        for white_plugin in args.plugins:

            print(white_plugin.lower())
            for plugin in plugins:
                if plugin.lower() == white_plugin.lower():
                    filtered_plugins.append(plugin)
        plugins = filtered_plugins

    print(plugins)
    indexing(args.path, args.output, args.batch, plugins, config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
