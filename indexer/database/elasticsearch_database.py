from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import exceptions
# from elasticsearch_dsl import Search

from indexer.database.database import Database


class ElasticSearchDatabase(Database):

    def __init__(self, config):
        self._es = Elasticsearch()
        self._index = 'iart'
        self._type = 'image'
        #
        # res = self._es.index(index="test-index", doc_type='tweet', id=1, body=doc)
        # print(res['result'])
        #
        # res = self._es.get(index="test-index", doc_type='tweet', id=1)
        # print(res['_source'])
        #
        # self._es.indices.refresh(index="test-index")
        #
        # res = self._es.search(index="test-index", body={"query": {"match_all": {}}})
        # print("Got %d Hits:" % res['hits']['total']['value'])
        # for hit in res['hits']['hits']:
        #     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

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
        print(entry)
        self._es.index(index=self._index, doc_type=self._type, id=hash_id, body=entry)

    # def search(self, query, size=10):
    #
    #     try:
    #
    #         doc = {'query': {'match_all': {}}}
    #         for x in self._es.search(index=self._index, body=doc, size=size)['hits']['hits']:
    #             yield x['_source']
    #     except exceptions.NotFoundError:
    #         return []

    def search(self, query, size=10):

        try:

            doc = query
            print(doc)
            for x in self._es.search(index=self._index, body=doc, size=size)['hits']['hits']:
                yield x['_source']
        except exceptions.NotFoundError:
            return []
        # self._es.update('')

    def drop(self):
        self._es.indices.delete(index=self._index, ignore=[400, 404])
