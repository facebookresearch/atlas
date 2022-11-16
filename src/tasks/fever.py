# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.evaluation import exact_match_score
from src.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["accuracy"]

    def process(self, example, *args, **kwargs):
        clean_input = example["claim"]

        clean_target = ""
        if "label" in example:
            target = example["label"]
            if target == "NOT ENOUGH INFO":
                clean_target = "maybe"
            elif target == "REFUTES":
                clean_target = "false"
            elif target == "SUPPORTS":
                clean_target = "true"

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = f"question: {clean_input} answer: <extra_id_0>"
        if clean_target is not None:
            example["target"] = f"<extra_id_0> {clean_target}"
        example["passages"] = [{"title": "", "text": ""}]
        example["metadata"]["clean_target"] = clean_target
        example["answers"] = [clean_target]

        return example

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {"accuracy": exact_match_score(prediction, ground_truths)}
        return sample_metrics
