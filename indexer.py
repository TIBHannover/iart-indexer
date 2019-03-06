import os
import sys
import re
import argparse
import logging

from plugins import *


def parse_args():
    parser = argparse.ArgumentParser(description='Indexing a set of images')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-p', '--path', required=True, help='verbose output')
    args = parser.parse_args()
    return args


def indexing(paths):
    find_plugins()
    if not isinstance(paths, (list, set)):
        paths = [paths]

    for plugin_name, plugin_class in feature_plugins().items():
        plugin = plugin_class()
        features = plugin(paths)


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)
    indexing(args.path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
