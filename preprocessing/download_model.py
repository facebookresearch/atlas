# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from download_tools import get_download_path, get_s3_path, maybe_download_file

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"
MODEL_FILE_NAME = "model.pth.tar"

AVAILABLE_MODELS = [
    {"model": "models/atlas/xxl", "description": "Pretrained Atlas XXL model"},
    {"model": "models/atlas/xl", "description": "Pretrained Atlas XL model"},
    {"model": "models/atlas/large", "description": "Pretrained Atlas Large model"},
    {"model": "models/atlas/base", "description": "Pretrained Atlas Base model"},
    {"model": "models/atlas_nq/xxl", "description": "Atlas XXL model, finetuned on Natural Questions"},
    {"model": "models/atlas_nq/xl", "description": "Atlas XL model, finetuned on Natural Questions"},
    {"model": "models/atlas_nq/large", "description": "Atlas large model, finetuned on Natural Questions"},
    {"model": "models/atlas_nq/base", "description": "Atlas base model, finetuned on Natural Questions"},
]


def _helpstr():
    helpstr = "The following models are available for download: "
    for m in AVAILABLE_MODELS:
        helpstr += f'\nModel name: {m["model"]:<30} Description: {m["description"]}'
    helpstr += "\ndownload by passing --model {model name}"
    return helpstr


def main(output_directory, requested_model):
    model_path = f"{requested_model}/{MODEL_FILE_NAME}"
    source = get_s3_path(model_path)
    target = get_download_path(output_directory, model_path)
    maybe_download_file(source, target)


if __name__ == "__main__":
    help_str = _helpstr()
    choices = list([a["model"] for a in AVAILABLE_MODELS])
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=choices,
        help=help_str,
    )
    args = parser.parse_args()
    main(args.output_directory, args.model)
