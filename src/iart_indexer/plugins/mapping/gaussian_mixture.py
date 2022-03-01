import random
import logging

import numpy as np

from sklearn.mixture import GaussianMixture

from iart_indexer.plugins import (
    MappingPlugin,
    MappingPluginManager,
    PluginResult,
)


@MappingPluginManager.export('GaussianMixtureMapping')
class GaussianMixtureMapping(MappingPlugin):
    default_config = {'random_state': 42, 'k': 5}
    default_version = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.random_state = self.config['random_state']
        self.default_k = self.config.get('k', 5)

        if self.random_state is None:
            self.random_state = random.randint()

    def call(self, entries, query, config: dict = None):
        k = self.default_k

        if isinstance(config, dict):
            try:
                k = int(config.get('k'))
            except:
                pass

        k = max(2, min(k, len(entries) - 1))
        ref_feature = 'clip_embedding_feature'

        logging.info(f'[GaussianMixtureMapping] k={k}')

        gm = GaussianMixture(
            n_clusters=k,
            random_state=self.random_state,
            init_params='kmeans',
        )

        features = []
        entries = list(entries)

        for e in entries:
            for e_f in e['feature']:
                if ref_feature != e_f['plugin']:
                    continue

                if 'value' in e_f['annotations'][0]:
                    a = e_f['annotations'][0]['value']

                features.append(a)

        features = np.asarray(features)
        clustering = gm.fit_transform(features)

        new_entries = [
            {
                **e,
                'cluster': gm.labels_[i].item(),
                'distance': clustering[i, gm.labels_[i].item()].item(),
            }
            for i, e in enumerate(entries)
        ]

        # sort in clusters, then distance from cluster center
        new_entries.sort(key=lambda x: x['distance'])
        new_entries.sort(key=lambda x: x['cluster'])

        return new_entries
