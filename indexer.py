import os
import sys
import re
import json
import argparse
import logging
import uuid
import imageio

from indexer.plugins import *
from indexer.config import IndexerConfig

from database.elasticsearch_database import ElasticSearchDatabase

from indexer.utils import copy_image_hash, filename_without_ext, image_resolution


def parse_args():
    parser = argparse.ArgumentParser(description='Indexing a set of images')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    # parser.add_argument('-l', '--list', help='list all plugins')
    parser.add_argument('-p', '--path', required=True, help='path to image or folder to indexing')
    parser.add_argument('-o', '--output', help='copy image to new folder with hash id')
    parser.add_argument('-a', '--append', action='store_true', help='add all found documents to the index')
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


def list_images(paths):
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
        'id': uuid.uuid4().hex,
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


def indexing(paths, output):

    # TODO replace with abstract class
    database = ElasticSearchDatabase(config=None)

    find_plugins()

    # handel images or jsonl
    if not isinstance(paths, (list, set)) and os.path.splitext(paths)[1] == '.jsonl':
        entries = list_jsonl(paths)
        print('test')
    else:
        entries = list_images(paths)
        print('test2')
    print(entries[:3])

    if output:
        entries = copy_images(entries, output)

    # TODO scroll
    existing_hash = [x['id'] for x in list(database.search(None))]
    for x in entries:
        if x['id'] not in existing_hash:
            resolution = image_resolution(x['path'])
            if resolution is None:
                continue
<<<<<<< HEAD
            print(x['meta'])
=======
>>>>>>> Add a script to add autocompletion index.
            database.insert_entry(
                x['id'], {
                    'id': x['id'],
                    'path': x['path'],
                    'filename': x['filename'],
<<<<<<< HEAD
                    'meta': x['meta'],
=======
>>>>>>> Add a script to add autocompletion index.
                    'image': {
                        'height': resolution[0],
                        'width': resolution[1]
                    }
                })

    logging.info(f'Indexing {len(entries)} documents')

    for plugin_name, plugin_class in feature_plugins().items():
        plugin = plugin_class()
        features = plugin(entries)

    for plugin_name, plugin_class in classifier_plugins().items():
        plugin = plugin_class()
        entries_processed = plugin(entries)
        print(entries_processed)
        update_database(database, entries_processed)


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)
    indexing(args.path, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
