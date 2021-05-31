import os
import sys
import re
import argparse
import faiss
import msgpack
import logging


def load_index(indexes_dir, index):
    if isinstance(index, dict):
        index_id = index["id"]
    elif isinstance(index, str):
        index_id = index
    else:
        raise KeyError
    with open(os.path.join(indexes_dir, index_id + ".msg"), "rb") as f:
        index = msgpack.unpackb(f.read())

    index["index"] = faiss.read_index(os.path.join(indexes_dir, index_id + ".index"))
    return index


def load_collection(collections_dir, indexes_dir, collection):
    if isinstance(collection, dict):
        collection_id = collection["id"]
    elif isinstance(collection, str):
        collection_id = collection
    else:
        raise KeyError

    with open(os.path.join(collections_dir, collection_id + ".msg"), "rb") as f:
        collection = msgpack.unpackb(f.read())

    indexes = []
    for index in collection["indexes"]:
        indexes.append(load_index(indexes_dir, index))
    collection["indexes"] = indexes

    return collection


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-p", "--path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if os.path.isdir(args.path):
        indexers = [os.path.join(args.path, x) for x in os.listdir(args.path) if re.match(r"^.*?\.msg$", x)]
        collections_dir = os.path.join(args.path, "collections")
        indexes_dir = os.path.join(args.path, "indexes")

    else:
        indexers = [args.path]
        collections_dir = os.path.join(os.path.dirname(args.path), "collections")
        indexes_dir = os.path.join(os.path.dirname(args.path), "indexes")

    newest_index = {"timestamp": 0.0}
    for x in indexers:
        with open(x, "rb") as f:
            data = msgpack.unpackb(f.read())
            if newest_index["timestamp"] < data["timestamp"]:
                newest_index = data

    print("####################")

    print()
    if "trained_collection" in newest_index:
        print("trained_collections: " + newest_index["trained_collection"])
        trained_collection = load_collection(collections_dir, indexes_dir, newest_index["trained_collection"])
        for index in trained_collection["indexes"]:
            print("\t" + index["id"] + " " + str(len(index["entries"])))
        print("default_collection: " + newest_index["default_collection"])
        # default_collection = load_collection(collections_dir, indexes_dir, newest_index["default_collection"])
        for collection in newest_index["collections"]:
            try:
                print("collection: " + str(collection))
                collection = load_collection(collections_dir, indexes_dir, collection)
                print(collection.keys())
                # print("collection: " + str(]) + " " + str(collection["timestamp"]))
                for index in collection["indexes"]:
                    print(index.keys())
                    print("\t\t" + index["id"] + " " + str(len(index["entries"])))
            except Exception as e:
                print(e)

    return 0


if __name__ == "__main__":
    sys.exit(main())