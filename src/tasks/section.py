# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from src.evaluation import exact_match_score, f1_score, rouge_score
from src.options import Options
from src.tasks.base import BaseTask, filter_results_by_id

logger = logging.getLogger(__name__)


class Task(BaseTask):

    metrics = ["eval_loss", "accuracy", "f1", "rouge_1", "rouge_2", "rouge_L"]

    def __init__(self, opt: Options, *args, **kwargs):
        self.min_words = opt.min_words_per_lm_instance

    def process(self, example, *args, **kwargs):

        if not "section" in example or len(example["section"].strip()) == 0:
            return

        query = ", ".join([example["title"], example["section"]])
        text = example["text"]
        if len(text.strip()) == 0:
            return
        if self.min_words is not None and len(text.split()) < self.min_words:
            return

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["query"] = query
        example["target"] = text
        example["metadata"] = {}
        example["metadata"]["id"] = example["id"]
        return example

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {}
        sample_metrics["accuracy"] = exact_match_score(prediction, ground_truths)
        sample_metrics["f1"] = f1_score(prediction, ground_truths)
        rouge_1, rouge_2, rouge_L = rouge_score(prediction, ground_truths)
        sample_metrics["rouge_1"] = rouge_1
        sample_metrics["rouge_2"] = rouge_2
        sample_metrics["rouge_L"] = rouge_L
        return sample_metrics

    def filter(self, *args, **kwargs):
        """Remove the passage we are trying to generate from retrieved results"""
        return filter_results_by_id(*args, **kwargs)
