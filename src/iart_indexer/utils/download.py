import os
import re
import sys
import time
import imageio
import logging
import urllib.error
import urllib.request

from tqdm import tqdm
from multiprocessing import Pool

from .image import image_resize


def download_image(url, max_dim=1024, try_count=2):
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
                image = image_resize(image, max_dim=max_dim)

                return image

        except urllib.error.URLError:
            time.sleep(1.0)
        except urllib.error.HTTPError:
            time.sleep(10.0)
        except KeyboardInterrupt:
            raise
        except Exception:
            time.sleep(1.0)

        try_count -= 1

    return None


def download_entry(
    entry,
    image_output,
    resolutions=[{"min_dim": 200, "suffix": "_m"}, {"suffix": ""}]
):
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
            new_image = image_resize(image, min_dim=res["min_dim"])
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

    with Pool(40) as p:
        values = [(e, image_output, resolutions) for e in entries]

        for i, x in enumerate(
            tqdm(
                p.imap(_download_entry, values),
                desc="Downloading", total=len(entries),
            )
        ):
            if i % 100 == 0:
                logging.info(f"Downloading {i}/{len(entries)}")

            if x is None:
                continue

            new_entries.append(x)

    return new_entries
