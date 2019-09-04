import os
import sys
import re
import argparse

from database.elasticsearch_suggester import ElasticSearchSuggester
from database.elasticsearch_database import ElasticSearchDatabase


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-b', '--build', action='store_true', help='verbose output')
    parser.add_argument('-q', '--query', help='verbose output')
    args = parser.parse_args()
    return args


def build_autocompletion():
    database = ElasticSearchDatabase()
    suggester = ElasticSearchSuggester()

    for x in database.all():
        print(x)
        meta_values = []
        if 'meta' in x:
            for key, value in x['meta'].items():
                if isinstance(value, str):
                    meta_values.append(value)
        annotations_values = []
        if 'classifier' in x:
            for classifier in x['classifier']:
                for annotations in classifier['annotations']:
                    annotations_values.append(annotations['name'])
        print(x['id'])
        suggester.update_entry(hash_id=x['id'], meta=meta_values, annotations=annotations_values)


def search(query):

    suggester = ElasticSearchSuggester()
    print(suggester.complete(query))


def main():
    args = parse_args()
    if args.build:
        build_autocompletion()

    if args.query is not None:
        search(args.query)

    return 0


if __name__ == '__main__':
    sys.exit(main())
