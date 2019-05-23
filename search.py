import os
import sys
import re
import argparse
import logging
import uuid
import json

from indexer.plugins import *
from indexer.config import IndexerConfig

from database.elasticsearch_database import ElasticSearchDatabase

from indexer.utils import copy_image_hash, filename_without_ext


def parse_args():
    parser = argparse.ArgumentParser(description='Indexing a set of images')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-q', '--query', help='list all plugins')
    parser.add_argument('-c', '--config', help='config path')
    parser.add_argument('-m', '--mode', choices=['local', 'server'], default='local', help='verbose output')
    args = parser.parse_args()
    return args


def search(query):
    print(query)

    database = ElasticSearchDatabase(config=None)
    for x in database.search(query):
        print(x)


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    query = None
    if args.query is not None:
        query = json.loads(args.query)
    search(query)
    return 0


if __name__ == '__main__':
    sys.exit(main())
