from indexer.plugins import FeaturePlugin
from indexer.plugins import FeaturePluginManager
from indexer.plugins import PluginResult
import tensorflow as tf
import numpy as np
import math

tf.enable_eager_execution()


@FeaturePluginManager.export('ColorHistogramFeature')
class ColorHistogramFeature(FeaturePlugin):

    def __init__(self, **kwargs):
        super(ColorHistogramFeature, self).__init__(**kwargs)

    def call(self, entries):
        result_entries = []
        result_annotations = []
        for entry in entries:
            entry_annotation = []

            image_data = tf.io.read_file(entry['path'])
            image = tf.image.decode_image(image_data, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image_yuv = tf.image.rgb_to_yuv(image)

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
            entry_annotation.append({'type': 'color', 'hash': hash_splits_dict, 'value': uv_histogram_norm.tolist()})

            result_annotations.append(entry_annotation)
            result_entries.append(entry)

        return PluginResult(self, result_entries, result_annotations)
