# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import re

from src.evaluation import exact_match_score, f1_score, rouge_score
from src.options import Options
from src.tasks.base import BaseTask, filter_results_by_id

logger = logging.getLogger(__name__)


class Task(BaseTask):
    metrics = ["eval_loss", "accuracy", "f1", "rouge_1", "rouge_2", "rouge_L"]

    def __init__(self, opt: Options, *args, **kwargs):
        self.min_words = opt.min_words_per_lm_instance
        self.min_context_ratio = opt.min_lm_context_ratio
        self.max_context_ratio = opt.max_lm_context_ratio

    def filter(self, *args, **kwargs):
        """Remove the passage we are trying to generate from retrieved results"""
        return filter_results_by_id(*args, **kwargs)

    def process(self, example, *args, **kwargs):

        text = example["text"]
        if len(text.strip()) == 0:
            return
        if self.min_words is not None and len(text.split()) < self.min_words:
            return
        inp, out = self.split(text, self.min_context_ratio, self.max_context_ratio)

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["query"] = inp
        example["target"] = out
        example["metadata"] = {}
        example["metadata"]["id"] = example["id"]
        return example

    @staticmethod
    def split(text, min_context_ratio, max_context_ratio):
        """Splits text into two segments for langauge modelling.
        Left segment is conditioning context, right segment is for generating.
        The left segment must be between min_context_ratio and max_context_ratio of right segement in terms of length.
        """
        words = re.split(r"(\S+)", text)
        min_length = int(max(2, len(words) * min_context_ratio))
        max_length = int(max(min(len(words) - 2, len(words) * max_context_ratio), min_length + 1))
        split_idx = random.randint(min_length, max_length)
        inp = "".join(words[:split_idx])
        out = "".join(words[split_idx:])
        return inp, out

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {}
        sample_metrics["accuracy"] = exact_match_score(prediction, ground_truths)
        sample_metrics["f1"] = f1_score(prediction, ground_truths)
        rouge_1, rouge_2, rouge_L = rouge_score(prediction, ground_truths)
        sample_metrics["rouge_1"] = rouge_1
        sample_metrics["rouge_2"] = rouge_2
        sample_metrics["rouge_L"] = rouge_L
        return sample_metrics
