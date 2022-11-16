# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.evaluation import exact_match_score, f1_score, rouge_score
from src.options import Options
from src.tasks.base import BaseTask, filter_results_by_id


class Task(BaseTask):
    metrics = ["eval_loss", "accuracy", "f1", "rouge_1", "rouge_2", "rouge_L"]

    def __init__(self, opt: Options, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        self.tokenizer = tokenizer
        self.min_words = opt.min_words_per_lm_instance
        self.mlm_noise_density = opt.mlm_noise_density
        self.mlm_mean_noise_span_length = opt.mlm_mean_noise_span_length
        self.text_maxlength = opt.text_maxlength

    def filter(self, *args, **kwargs):
        """Remove the passage we are trying to denoise from retrieved results"""
        return filter_results_by_id(*args, **kwargs)

    def process(self, example, *args, **kwargs):
        """Noises the target field using T5 MLM masking, saves the orginal target in metadata,"""

        clean_target = example["text"]
        if len(clean_target.strip()) == 0:
            return None
        if self.min_words is not None and len(clean_target.split()) < self.min_words:
            return None

        output_example = {}

        inp, out = self.apply_mlm_noise(
            self.tokenizer,
            clean_target,
            self.mlm_noise_density,
            self.mlm_mean_noise_span_length,
            self.text_maxlength,
        )
        if not "passages" in example:
            output_example["passages"] = [{"title": "", "text": ""}]

        output_example["query"] = inp
        output_example["target"] = out
        output_example["metadata"] = example
        output_example["metadata"]["clean_target"] = clean_target
        return output_example

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {}
        sample_metrics["accuracy"] = exact_match_score(prediction, ground_truths)
        sample_metrics["f1"] = f1_score(prediction, ground_truths)
        rouge_1, rouge_2, rouge_L = rouge_score(prediction, ground_truths)
        sample_metrics["rouge_1"] = rouge_1
        sample_metrics["rouge_2"] = rouge_2
        sample_metrics["rouge_L"] = rouge_L
        return sample_metrics

    @staticmethod
    def apply_mlm_noise(
        tokenizer,
        text,
        mlm_noise_density,
        mlm_mean_noise_span_length,
        max_input_length,
    ):

        tokens = tokenizer(text, add_special_tokens=False, max_length=max_input_length, truncation=True)["input_ids"]
        length = len(tokens)

        num_noise_tokens = max(round(length * mlm_noise_density), 1)
        num_noise_spans = max(round(num_noise_tokens / mlm_mean_noise_span_length), 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def _get_span_lengths(num_items, num_segments):
            positions = [i < (num_segments - 1) for i in range(num_items - 1)]
            random.shuffle(positions)
            positions.append(True)
            output, prev_span_start = [], -1
            for i, n in enumerate(positions):
                if n:
                    output.append(i - prev_span_start)
                    prev_span_start = i
            return output

        noise_span_lengths = _get_span_lengths(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _get_span_lengths(num_nonnoise_tokens, num_noise_spans)

        inputs, outputs, offset = [], [], 0
        for i, (inp_length, out_length) in enumerate(zip(nonnoise_span_lengths, noise_span_lengths)):
            sentinel_id = tokenizer.additional_special_tokens_ids[i]
            inputs += tokens[offset : offset + inp_length] + [sentinel_id]
            offset += inp_length
            outputs += [sentinel_id] + tokens[offset : offset + out_length]
            offset += out_length

        return tokenizer.decode(inputs), tokenizer.decode(outputs)
