# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from pathlib import Path
import argparse

from download_tools import maybe_download_file

fever_64shot = [
    23236,
    131610,
    70907,
    110333,
    83874,
    121714,
    17355,
    115320,
    9907,
    42725,
    43614,
    139489,
    30589,
    76963,
    5916,
    7241,
    68848,
    59902,
    113855,
    110886,
    102332,
    79223,
    24359,
    105929,
    131435,
    118883,
    8152,
    119911,
    28803,
    111318,
    29503,
    43420,
    39533,
    15214,
    29807,
    29242,
    10288,
    111860,
    77451,
    102160,
    77982,
    132435,
    2875,
    47721,
    92378,
    128574,
    24721,
    83985,
    41521,
    97851,
    137243,
    74916,
    85056,
    135,
    130085,
    19233,
    2887,
    124345,
    91769,
    63969,
    50865,
    135928,
    143220,
    124300,
]


def main(args):
    output_dir = Path(args.output_directory)

    fever_path, fever_url = {}, {}
    fever_dir = output_dir / "fever_data"
    fever_path["train"] = fever_dir / "train.jsonl"
    fever_path["train-64"] = fever_dir / "train-64.jsonl"
    fever_path["dev"] = fever_dir / "dev.jsonl"
    fever_path["test"] = fever_dir / "test.jsonl"

    fever_url["train"] = "https://fever.ai/download/fever/train.jsonl"
    fever_url["dev"] = "https://fever.ai/download/fever/shared_task_dev.jsonl"
    fever_url["test"] = "https://fever.ai/download/fever/shared_task_test.jsonl"

    for split in ["train", "dev", "test"]:
        if args.overwrite or not fever_path[split].exists():
            maybe_download_file(fever_url[split], fever_path[split])
        else:
            print(f"{split} file already exists, not overwriting, use --overwrite instead")

    with open(fever_path["train"]) as fin:
        with open(fever_path["train-64"], "w") as fout:
            for k, line in enumerate(fin):
                if k in fever_64shot:
                    ex = json.loads(line)
                    json.dump(ex, fout)
                    fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data/",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite data")
    args = parser.parse_args()
    main(args)
