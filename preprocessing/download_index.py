# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from download_tools import get_download_path, get_s3_path, maybe_download_file

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"
PASSAGE_FNAME = "passages.{shard}.pt"
EMBEDDING_FNAME = "embeddings.{shard}.pt"
N_SHARDS = 128
AVAILABLE_INDICES = [
    {
        "index": "indices/atlas/wiki/xxl",
        "description": "Precomputed index for the wiki-dec2018 corpus for the pretrained atlas xxl model",
    },
    {
        "index": "indices/atlas/wiki/xl",
        "description": "Precomputed index for the wiki-dec2018 corpus for the pretrained atlas xl model",
    },
    {
        "index": "indices/atlas/wiki/large",
        "description": "Precomputed index for the wiki-dec2018 corpus for the pretrained atlas large model",
    },
    {
        "index": "indices/atlas/wiki/base",
        "description": "Precomputed index for the wiki-dec2018 corpus for the pretrained atlas base model",
    },
    {
        "index": "indices/atlas_nq/wiki/xxl",
        "description": "Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned atlas xxl model",
    },
    {
        "index": "indices/atlas_nq/wiki/xl",
        "description": "Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned atlas xl model",
    },
    {
        "index": "indices/atlas_nq/wiki/large",
        "description": "Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned atlas large model",
    },
    {
        "index": "indices/atlas_nq/wiki/base",
        "description": "Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned atlas base model",
    },
]


def _helpstr():
    helpstr = "The following indices are available for download: "
    for m in AVAILABLE_INDICES:
        helpstr += f'\nIndex name: {m["index"]:<30} Description: {m["description"]}'
    helpstr += "\nDownload by passing --index {index name}"
    return helpstr


def get_passage_path(index, shard_number):
    passage_filename = PASSAGE_FNAME.format(shard=shard_number)
    return f"{index}/{passage_filename}"


def get_embedding_path(index, shard_number):
    embedding_filename = EMBEDDING_FNAME.format(shard=shard_number)
    return f"{index}/{embedding_filename}"


def main(output_directory, requested_index):
    for shard in range(N_SHARDS):
        passage_path = get_passage_path(requested_index, shard)
        source = get_s3_path(passage_path)
        target = get_download_path(output_directory, passage_path)
        maybe_download_file(source, target)

        embedding_path = get_embedding_path(requested_index, shard)
        source = get_s3_path(embedding_path)
        target = get_download_path(output_directory, embedding_path)
        maybe_download_file(source, target)


if __name__ == "__main__":
    help_str = _helpstr()
    choices = list([a["index"] for a in AVAILABLE_INDICES])
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument(
        "--index",
        type=str,
        choices=choices,
        help=help_str,
    )
    args = parser.parse_args()
    main(args.output_directory, args.index)
