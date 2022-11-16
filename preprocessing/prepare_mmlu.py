# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import os
import random
import tarfile
from pathlib import Path

from download_tools import maybe_download_file

DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def maybe_download_data(output_directory):
    os.makedirs(output_directory, exist_ok=True)

    # download tar:
    orig_data_tar = output_directory / "data.tar"
    maybe_download_file(DATA_URL, orig_data_tar)

    untarred_orig_data = Path(output_directory) / "data"
    if not os.path.exists(untarred_orig_data):
        with tarfile.open(orig_data_tar) as tar:
            tar.extractall(output_directory)

    return untarred_orig_data


def build_mmlu_instance(name, line):
    question, option_a, option_b, option_c, option_d, answer = line
    return {
        "question": question,
        "options": {"A": option_a, "B": option_b, "C": option_c, "D": option_d},
        "answer": answer,
        "dataset": name,
    }


def get_dataset_name_from_path(path):
    return os.path.basename(path).replace(".csv", "")


def parse_mmlu_csv(path):
    output = []
    name = get_dataset_name_from_path(path)

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line in csv_reader:
            obj = build_mmlu_instance(name, line)
            output.append(obj)

    return output


def parse_all_mmlu_data(directory):
    all_data = {}

    for split in ["auxiliary_train", "dev", "val", "test"]:
        for fi in os.listdir(directory / split):
            path_to_read = directory / split / fi
            name = get_dataset_name_from_path(path_to_read)
            all_data.setdefault(split, {})[name] = parse_mmlu_csv(path_to_read)

    return all_data


def dump(items, path):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def make_five_shot_data(data, output_directory):
    indiv_train_path = output_directory / "individual_train"
    indiv_valid_path = output_directory / "individual_valid"
    indiv_test_path = output_directory / "individual_test"
    os.makedirs(indiv_train_path, exist_ok=True)
    os.makedirs(indiv_valid_path, exist_ok=True)
    os.makedirs(indiv_test_path, exist_ok=True)

    for domain, items in data["dev"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_train_path / f"{domain}.5-shot-train.jsonl"
        dump(items, dump_path)

    for domain, items in data["val"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_valid_path / f"{domain}.val.jsonl"
        dump(items, dump_path)

    for domain, items in data["test"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_test_path / f"{domain}.test.jsonl"
        dump(items, dump_path)

    combined_val = [item for _, items in data["val"].items() for item in items]
    dump(combined_val, output_directory / f"combined_valid.jsonl")

    combined_test = [item for _, items in data["test"].items() for item in items]
    dump(combined_test, output_directory / f"combined_test.jsonl")


def make_five_shot_multitask_data(data, output_directory):
    indiv_valid_path = output_directory / "individual_valid"
    indiv_test_path = output_directory / "individual_test"
    os.makedirs(indiv_valid_path, exist_ok=True)
    os.makedirs(indiv_test_path, exist_ok=True)

    for domain, items in data["val"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_valid_path / f"{domain}.val.jsonl"
        dump(items, dump_path)

    for domain, items in data["test"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_test_path / f"{domain}.test.jsonl"
        dump(items, dump_path)

    combined_train = [item for _, items in data["dev"].items() for item in items]
    dump(combined_train, output_directory / f"train.jsonl")

    combined_val = [item for _, items in data["val"].items() for item in items]
    dump(combined_val, output_directory / f"combined_valid.jsonl")

    combined_test = [item for _, items in data["test"].items() for item in items]
    dump(combined_test, output_directory / f"combined_test.jsonl")


def make_full_transfer_data(data, output_directory):
    indiv_valid_path = output_directory / "individual_valid"
    indiv_test_path = output_directory / "individual_test"
    os.makedirs(indiv_valid_path, exist_ok=True)
    os.makedirs(indiv_test_path, exist_ok=True)

    for domain, items in data["val"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_valid_path / f"{domain}.val.jsonl"
        dump(items, dump_path)

    for domain, items in data["test"].items():
        domain = "_".join(domain.split("_")[:-1])
        dump_path = indiv_test_path / f"{domain}.test.jsonl"
        dump(items, dump_path)

    combined_auxilary = [item for _, items in data["auxiliary_train"].items() for item in items]
    random.seed(10)
    random.shuffle(combined_auxilary)
    auxillary_valid = combined_auxilary[-5000:]
    auxiliary_train = combined_auxilary[:-5000]

    dump(auxillary_valid, output_directory / f"auxillary_valid.jsonl")

    combined_train = [item for _, items in data["dev"].items() for item in items]
    full_train = auxiliary_train + combined_train
    dump(full_train, output_directory / f"train.jsonl")

    combined_val = [item for _, items in data["val"].items() for item in items]
    dump(combined_val, output_directory / f"combined_valid.jsonl")

    combined_test = [item for _, items in data["test"].items() for item in items]
    dump(combined_test, output_directory / f"combined_test.jsonl")


def main(output_directory):
    original_data_directory = maybe_download_data(output_directory)
    all_data = parse_all_mmlu_data(original_data_directory)

    make_five_shot_data(all_data, output_directory / "5-shot")
    make_five_shot_multitask_data(all_data, output_directory / "5-shot-multitask")
    make_full_transfer_data(all_data, output_directory / "full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Downloads, parses and creates train, validation and test files for MMLU.

We consider 3 tasks:
* 5-shot: learn a model with 5 examples for each domain. 
* 5-shot-multitask: Learn a single model using the combination of 5 examples from each domain.
* full: Learn a single model using training data from MMLU's auxialluary datasets, plus the training data from 5-shot-multitask.

In each case, overall test accuracy would be the micro average over each domains' test set (as defined by the orginal authors).

The script will download the data, and create the following directory structure:

├── data.tar # original data
├── 5-shot 
│   ├── combined_test.jsonl
│   ├── combined_valid.jsonl
│   ├── individual_test
│   │   ├── {domain}.test.jsonl
│   ├── individual_train
│   │   ├── {domain}.5-shot-train.jsonl
│   └── individual_valid
│       ├── {domain}.val.jsonl
├── 5-shot-multitask 
│   ├── combined_test.jsonl
│   ├── combined_valid.jsonl
│   ├── individual_test
│   │   ├── {domain}.test.jsonl
│   ├── individual_valid
│   │   ├── {domain}.val.jsonl
│   └── train.jsonl
└── full
    ├── auxillary_valid.jsonl
    ├── combined_test.jsonl
    ├── combined_valid.jsonl
    ├── individual_test
    │   ├── {domain}.test.jsonl
    ├── individual_valid
    │   ├── {domain}.val.jsonl
    └── train.jsonl

* For 5-shot,  train models 5-shot/individual_train/{domain}.5-shot-train.jsonl and test on 5-shot/individual_test/{domain}.test.jsonl
* For 5-shot-multitask, train models 5-shot-multitask/train.jsonl and test on 5-shot-multitask/combined_test.jsonl
* For the full data task, train models full/train.jsonl and test on full/combined_test.jsonl

"""
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data/",
        help="Path to the file to which the dataset is written.",
    )
    args = parser.parse_args()
    output_directory = Path(args.output_directory) / "data" / "mmlu_data"
    main(output_directory)
