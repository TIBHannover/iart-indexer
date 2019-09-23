from indexer.plugins import FeaturePlugin
from indexer.plugins import FeaturePluginManager
from indexer.plugins import PluginResult
import tensorflow as tf
import numpy as np
import math

tf.enable_eager_execution()


@FeaturePluginManager.export('ImageNetInceptionFeature')
class ImageNetInceptionFeature(FeaturePlugin):

    def __init__(self, **kwargs):
        super(ImageNetInceptionFeature, self).__init__(**kwargs)
        config = kwargs.get('config', {})

        if 'model_path' not in config:
            logging.error()
            return

        with gfile.FastGFile(config['model_path'], 'rb') as file:
            self._graph = tf.GraphDef()
            self._graph.ParseFromString(file.read())
            _ = tf.import_graph_def(self._graph, name='')

        if 'pca_path' not in config:
            logging.error()
            return

        with open(config['pca_path'], 'rb') as f:
            pca_params = pickle.load(f)

        print('TEst')
        exit()

    def dim_reduction(self, features):
        features = features - pca_params.mean_
        reduc_dim_features = np.dot(features, pca_params.components_.T)
        reduc_dim_features /= np.sqrt(pca_params.explained_variance_)
        return reduc_dim_features

    def call(self, entries):
        result_entries = []
        result_annotations = []

        with tf.Session() as sess:
            for entry in entries:
                entry_annotation = []

                conv_output_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                data = gfile.FastGFile(image_path, 'rb').read()
                try:
                    feature = sess.run(conv_output_tensor, {'DecodeJpeg/contents:0': data})
                    feature = np.squeeze(feature)
                except:
                    continue
                image_uv = image_yuv.numpy().reshape([-1, 3])
                uv_histogram, edges = np.histogramdd(image_uv, 4, [[0, 1], [-0.5, 0.5], [-0.5, 0.5]])
                # img = imageio.imread(entry['path'])
                uv_histogram_norm = uv_histogram / (image_uv.shape[0])
                uv_histogram_norm = uv_histogram_norm.flatten()
                uv_histogram_norm_bin = ''.join(
                    [str(int(x > 0)) for x in (uv_histogram_norm / np.mean(uv_histogram_norm)).tolist()])

                hash_splits_dict = {}
                for x in range(math.ceil(len(uv_histogram_norm_bin) / 16)):
                    # print(uv_histogram_norm_bin[x * 16:(x + 1) * 16])
                    hash_splits_dict[f'split_{x}'] = uv_histogram_norm_bin[x * 16:(x + 1) * 16]

                # TODO split yuv and lab color.rgb2lab
                entry_annotation.append({
                    'type': 'color',
                    'hash': hash_splits_dict,
                    'value': uv_histogram_norm.tolist()
                })

                result_annotations.append(entry_annotation)
                result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
