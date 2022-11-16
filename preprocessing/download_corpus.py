# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from download_tools import get_download_path, get_s3_path, maybe_download_file

AVAILABLE_CORPORA = {
    "corpora/wiki/enwiki-dec2017": {
        "corpus": "corpora/wiki/enwiki-dec2017",
        "description": "Wikipedia dump from Dec 2017, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2018": {
        "corpus": "corpora/wiki/enwiki-dec2018",
        "description": "Wikipedia dump from Dec 2018, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-aug2019": {
        "corpus": "corpora/wiki/enwiki-aug2019",
        "description": "Wikipedia dump from Aug 2019, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2020": {
        "corpus": "corpora/wiki/enwiki-dec2020",
        "description": "Wikipedia dump from Dec 2020, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2021": {
        "corpus": "corpora/wiki/enwiki-dec2021",
        "description": "Wikipedia dump from Dec 2021, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
}


def _helpstr():
    helpstr = "The following corpora are available for download: "
    for m in AVAILABLE_CORPORA.values():
        helpstr += f'\nCorpus name: {m["corpus"]:<30} Description: {m["description"]}'
    helpstr += "\ndownload by passing --corpus {corpus name}"
    return helpstr


def main(output_directory, requested_corpus):
    AVAILABLE_CORPORA[requested_corpus]
    for filename in AVAILABLE_CORPORA[requested_corpus]["files"]:
        path = f"{requested_corpus}/{filename}"
        source = get_s3_path(path)
        target = get_download_path(output_directory, path)
        maybe_download_file(source, target)


if __name__ == "__main__":
    help_str = _helpstr()
    choices = list(AVAILABLE_CORPORA.keys())
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=choices,
        help=help_str,
    )
    args = parser.parse_args()
    main(args.output_directory, args.corpus)
