# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from pathlib import Path

from download_tools import maybe_download_file

URLS = {
    "train": "https://storage.googleapis.com/gresearch/templama/train.json",
    "valid": "https://storage.googleapis.com/gresearch/templama/val.json",
    "test": "https://storage.googleapis.com/gresearch/templama/test.json",
}


def prep_question(question):
    return question.replace("_X_", "<extra_id_0>")


def maybe_download_data(output_directory):
    paths = {}
    for split, url in URLS.items():
        dest = output_directory / f"{split}.original.jsonl"
        maybe_download_file(url, dest)
        paths[split] = dest
    return paths


def _parse(path, years_to_parse):
    items = []
    for line in open(path):
        if line.strip() != "":
            items.append(json.loads(line))

    mapper = {}
    for i in items:
        if i["date"] in years_to_parse:
            mapper.setdefault(i["query"], []).append(i)

    return mapper


def _dump(output_path, objects_to_write):
    with open(output_path, "w") as f:
        for item in objects_to_write:
            f.write(json.dumps(item) + "\n")


def _get_export_obj(obj):
    return {
        "question": prep_question(obj["query"]),
        "answers": list(set([n["name"] for n in obj["answer"]])),
        "metadata": {"original_instance": obj},
    }


def main(output_directory, years_to_compare=["2017", "2020"]):
    os.makedirs(output_directory, exist_ok=True)
    paths = maybe_download_data(output_directory)

    for split, path in paths.items():

        to_write = {y: [] for y in years_to_compare}
        query2items = _parse(path, years_to_compare)

        for _, objects in query2items.items():
            if len(objects) == 1:  # question doesnt have different answers at different years
                continue

            first_answer, later_answers = objects[0], objects[1:]
            previous_answer_strings = set([n["name"] for n in first_answer["answer"]])

            different_later_answers = []
            for later_answer in later_answers:
                if all([n["name"] not in previous_answer_strings for n in later_answer["answer"]]):
                    different_later_answers.append(later_answer)

            if len(different_later_answers) > 0:
                to_write[first_answer["date"]].append(_get_export_obj(first_answer))
                for d in different_later_answers:
                    to_write[d["date"]].append(_get_export_obj(d))

        for date, items in to_write.items():
            output_path = output_directory / f"temp_lama.{split}.{date}.jsonl"
            _dump(output_path, items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Path to the file to which the dataset is written.",
    )
    args = parser.parse_args()
    output_directory = Path(args.output_directory) / "data" / "templama_data"
    main(output_directory)
