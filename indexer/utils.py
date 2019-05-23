import os
import sys
import re
import argparse
import uuid
import imageio
import struct


def convert_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def copy_image_hash(image_path, output_path):
    try:
        hash_value = uuid.uuid4().hex

        output_dir = os.path.join(output_path, hash_value[0:2], hash_value[2:4])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f'{hash_value}.jpg')

        image = imageio.imread(image_path)

        imageio.imwrite(output_file, image)

        return hash, output_file
    except ValueError as e:
        return None
    except struct.error as e:
        return None


def filename_without_ext(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]
