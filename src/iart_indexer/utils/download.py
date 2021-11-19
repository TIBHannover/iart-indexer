import os
import sys
import re
import argparse

import urllib.error
import urllib.request
import imageio
import hashlib
import time

import numpy as np
import multiprocessing as mp

import utils
import logging


def download_image(url, max_dim=1024, try_count=2):

    try_count = try_count
    while try_count > 0:
        try:
            request = urllib.request.Request(
                url=url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3)"
                        " AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/48.0.2564.116 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                image = imageio.imread(response.read(), pilmode="RGB")
                image = utils.image_resize(image, max_dim=max_dim)
                return image

        except urllib.error.URLError as err:
            time.sleep(1.0)
        except urllib.error.HTTPError as err:
            time.sleep(10.0)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            time.sleep(1.0)

        try_count -= 1

    return None


def download_entry(entry, image_output, resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}]):
    if os.path.splitext(entry["origin"]["link"])[1].lower()[1:] in [
        "svg",
        "djvu",
        "webm",
        "ogv",
        "gif",
        "pdf",
        "ogg",
        "oga",
        "mid",
    ]:
        return None

    logging.info(f'{entry["origin"]["link"]} {entry["id"]}')
    image = download_image(entry["origin"]["link"])

    if image is None:
        return None

    sample_id = entry["id"]
    hash_value = sample_id
    output_dir = os.path.join(image_output, sample_id[0:2], sample_id[2:4])
    os.makedirs(output_dir, exist_ok=True)

    for res in resolutions:
        if "min_dim" in res:
            new_image = utils.image_resize(image, min_dim=res["min_dim"])
            image_output_file = os.path.join(output_dir, f"{hash_value}{res['suffix']}.jpg")
        else:
            new_image = image
            image_output_file = os.path.join(output_dir, f"{hash_value}{res['suffix']}.jpg")

        imageio.imwrite(image_output_file, new_image)

    image_output_file = os.path.abspath(os.path.join(output_dir, f"{hash_value}.jpg"))

    return {**entry, "path": os.path.join(output_dir, f"{hash_value}.jpg")}


def _download_entry(args):
    return download_entry(*args)


def download_entries(
    entries,
    image_output=None,
    resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}],
):
    new_entries = []
    with mp.Pool(40) as p:

        for i, x in enumerate(p.imap(_download_entry, [(e, image_output, resolutions) for e in entries])):
            if i % 100 == 0:
                logging.info(f"Entries downloader: Downloading {i}/{len(entries)}")
            if x is None:
                continue

            print(f'Reading {x["path"]}')
            new_entries.append(x)

    return new_entries
