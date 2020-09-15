import os
import sys
import re
import argparse
import logging
import uuid

from indexer.indexer.plugins import *
from indexer.indexer.config import IndexerConfig

from indexer.database.elasticsearch_database import ElasticSearchDatabase

from indexer.indexer.utils import copy_image_hash, filename_without_ext


def parse_args():
    parser = argparse.ArgumentParser(description="Indexing a set of images")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    # parser.add_argument('-q', '--query', help='list all plugins')
    parser.add_argument("-c", "--config", help="config path")
    parser.add_argument("-m", "--mode", choices=["local", "server"], default="local", help="verbose output")
    args = parser.parse_args()
    return args


def delete():

    database = ElasticSearchDatabase(config=None)
    database.drop()


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)
    delete()
    return 0


if __name__ == "__main__":
    sys.exit(main())
