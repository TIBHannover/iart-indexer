import os
import sys
import re
import argparse
import uuid
import imageio
import struct
import skimage


def convert_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def copy_image_hash(image_path, output_path, hash_value=None, resolutions=[-1]):
    try:
        if hash_value is None:
            hash_value = uuid.uuid4().hex

        output_dir = os.path.join(output_path, hash_value[0:2], hash_value[2:4])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = imageio.imread(image_path)

        for res in resolutions:
            if res > 0:
                scale_factor = max(image.shape[0] / float(res), image.shape[1] / float(res))
                new_image = skimage.transform.rescale(image, 1 / scale_factor)

                output_file = os.path.join(output_dir, f'{hash_value}_{res}.jpg')
            else:
                new_image = image
                output_file = os.path.join(output_dir, f'{hash_value}.jpg')

            imageio.imwrite(output_file, new_image)

        output_file = os.path.abspath(os.path.join(output_dir, f'{hash_value}.jpg'))
        return hash, output_file
    except ValueError as e:
        return None
    except struct.error as e:
        return None


def image_resolution(image_path):
    try:
        image = imageio.imread(image_path)

        return image.shape[0], image.shape[1]
    except ValueError as e:
        return None
    except struct.error as e:
        return None


def filename_without_ext(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]
