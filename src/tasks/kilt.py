# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List

from src.evaluation import exact_match_score, f1_score, normalize_answer
from src.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["accuracy", "exact_match", "f1"]

    def process(self, example, *args, **kwargs):

        clean_input = example["input"]

        answers = list(self.get_gold_answers(example))
        if "filename" in example and "fever" in example["filename"]:
            answers = ["true" if a == "SUPPORTS" else "false" for a in answers]
        clean_target = random.choice(answers)

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = f"question: {clean_input} answer: <extra_id_0>"
        example["target"] = f"<extra_id_0> {clean_target}"
        example["answers"] = answers
        example["passages"] = [{"title": "", "text": ""}]
        example["metadata"]["clean_target"] = clean_target

        return example

    def get_gold_answers(self, gold):
        ground_truths = set()
        for item in gold["output"]:
            if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
                ground_truths.add(item["answer"].strip())
        return ground_truths

    def evaluation(self, prediction: str, ground_truths: List[str]):
        sample_metrics = {
            "accuracy": exact_match_score(prediction, ground_truths),
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
