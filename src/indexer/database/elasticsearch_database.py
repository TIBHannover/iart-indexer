from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import exceptions
# from elasticsearch_dsl import Search

from database.database import Database


class ElasticSearchDatabase(Database):

    def __init__(self, config=None):
        self._es = Elasticsearch()
        self._index = 'iart'
        self._type = '_doc'

    def insert_entry(self, hash_id, doc):
        self._es.index(index=self._index, doc_type=self._type, id=hash_id, body=doc)

    def update_entry(self, hash_id, doc):
        self._es.update(index=self._index, doc_type=self._type, id=hash_id, body={'doc': doc})

    def get_entry(self, hash_id):
        return self._es.get(index=self._index, doc_type=self._type, id=hash_id)['_source']

    def update_plugin(self, hash_id, plugin_name, plugin_version, plugin_type, annotations):
        entry = self.get_entry(hash_id=hash_id)
        if entry is None:
            # TODO logging
            return

        if plugin_type in entry:
            for i, plugin in enumerate(entry[plugin_type]):
                if plugin['plugin'] == plugin_name and plugin['version'] < plugin_version:
                    entry[plugin_type][i] = {
                        'plugin': plugin_name,
                        'version': plugin_version,
                        'annotations': annotations
                    }

        else:
            entry.update(
                {plugin_type: [{
                    'plugin': plugin_name,
                    'version': plugin_version,
                    'annotations': annotations
                }]})
        # exit()
        self._es.index(index=self._index, doc_type=self._type, id=hash_id, body=entry)

    def search(self, meta=None, features=None, annotations=None, sort=None, size=5):

        body = {}

        if meta is not None:
            body.update({
                'query': {
                    'multi_match': {
                        'query': meta,
                        'fields': ['meta.title', 'meta.author', 'meta.location', 'meta.institution']
                    }
                }
            })

        if annotations is not None:
            body.update({'query': {'match': {'classifier.annotations.name': annotations}}})

        if sort is not None:
            sort_list = []
            if not isinstance(sort, (list, set)):
                sort = [sort]

            for x in sort:
                if x == 'annotations':
                    sort_list.append({'classifier.annotations.value': 'desc'})

            body.update({'sort': sort_list})
        print(body)
        try:
            results = self._es.search(index=self._index, body=body, size=size)
            print(results)
            for x in results['hits']['hits']:
                yield x['_source']
        except exceptions.NotFoundError:
            return []
        # self._es.update('')

    def all(self, pagesize=250, scroll_timeout="10m", **kwargs):
        is_first = True
        while True:
            # Scroll next
            if is_first:  # Initialize scroll
                result = self._es.search(index=self._index, scroll="1m", **kwargs, body={"size": pagesize})
                is_first = False
            else:
                result = self._es.scroll(body={"scroll_id": scroll_id, "scroll": scroll_timeout})
            scroll_id = result["_scroll_id"]
            hits = result["hits"]["hits"]
            # Stop after no more docs
            if not hits:
                break
            # Yield each entry
            yield from (hit['_source'] for hit in hits)

    def drop(self):
        self._es.indices.delete(index=self._index, ignore=[400, 404])
