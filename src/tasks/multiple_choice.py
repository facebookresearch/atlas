# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import string

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.evaluation import exact_match_score
from src.options import Options
from src.tasks.base import BaseTask


def _get_permutation_orderings(N, permutations_type):
    li = list(range(N))
    if permutations_type == "cyclic":
        orderings = [li[N - i :] + li[: N - i] for i in range(N)]
    elif permutations_type == "all":
        orderings = list(itertools.permutations(li))
    else:
        orderings = [li]
    return orderings


class Task(BaseTask):
    metrics = ["debiased_accuracy", "accuracy", "eval_loss"]

    def __init__(self, opt: Options, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.maximum_question_length = 356
        self.choices = string.ascii_uppercase[: opt.multiple_choice_num_options]
        self.choice2index = {o: self.tokenizer(o)["input_ids"][0] for o in self.choices}

    @staticmethod
    def get_multiple_choice_question_prompt(tokenizer, question, choices, maximum_length=356):
        def _length_in_tokens(string):
            return len(tokenizer(string)["input_ids"])

        def _get_prompt(question, choices_wseparator):
            preprocessed_question = f"question: {question.strip()} options: {choices_wseparator} answer: <extra_id_0>"
            return preprocessed_question

        choices_wseparator = " ".join([f"({L}) {T}" for L, T in choices.items()]).strip()
        question_with_options = _get_prompt(question, choices_wseparator)

        if _length_in_tokens(question_with_options) > maximum_length:
            max_qlen = maximum_length - _length_in_tokens(_get_prompt("", choices_wseparator))
            truncated_question = tokenizer.decode(
                tokenizer(question)["input_ids"][-max_qlen:], skip_special_tokens=True
            )
            question_with_options = _get_prompt(truncated_question, choices_wseparator)

        return question_with_options

    def process(self, example, *args, **kwargs):
        preprocessed_question = self.get_multiple_choice_question_prompt(
            self.tokenizer, example["question"], example["options"], maximum_length=self.maximum_question_length
        )
        target = f'<extra_id_0> {example["answer"]}'

        return {
            "query": preprocessed_question,
            "target": target,
            "choices": self.choices,
            "passages": [{"title": "", "text": ""}],
            "answers": [example["answer"]],
            "metadata": example,
        }

    @staticmethod
    def get_permutations(example, permutations_type):
        """clones example according to permutations_type (either "none", 'cyclic' or 'full'"""
        options, answer = example["options"], example["answer"]
        uid = example["question"] + " ".join(options.values())

        choice_keys = list(sorted(options.keys()))
        choice_values = [options[l] for l in choice_keys]
        orderings = _get_permutation_orderings(len(choice_keys), permutations_type)

        permuted_examples = []
        for ordering in orderings:
            permuted_options = {l: choice_values[o] for l, o in zip(choice_keys, ordering)}
            permuted_answer = [k for k, ans in permuted_options.items() if ans == options[answer]][0]

            permed_example = copy.deepcopy(example)
            permed_example["options"] = permuted_options
            permed_example["answer"] = permuted_answer
            permed_example["is_original"] = permuted_options == example["options"]
            permed_example["uid"] = uid
            permuted_examples.append(permed_example)

        return permuted_examples

    @staticmethod
    def data_iterator(*args, **kwargs):
        # wrap base data iterator in the case of permuting examples
        super_iterator = super(Task, Task).data_iterator(*args, **kwargs)
        perms_type = (
            kwargs["opt"].multiple_choice_eval_permutations
            if kwargs.get("is_eval", False)
            else kwargs["opt"].multiple_choice_train_permutations
        )
        for example in super_iterator:
            for permed_item in Task.get_permutations(example, perms_type):
                yield permed_item

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {"accuracy": exact_match_score(prediction, ground_truths)}
        return sample_metrics

    def get_choice_logits(self, logits):
        prediction_logits = {
            letter: logits[1, letter_index].cpu().item() for letter, letter_index in self.choice2index.items()
        }
        return prediction_logits

    def _get_original_instance(self, permutations):
        return [p for p in permutations if p["metadata"]["is_original"]][0]

    def _marginalize_across_permutations(self, permutations):
        original_instance = self._get_original_instance(permutations)
        text_answer_2_letter = {v: k for k, v in original_instance["metadata"]["options"].items()}

        aggregate_probs = {}
        for perm in permutations:
            logits = torch.tensor([perm["choice_logits"][c] for c in self.choices])
            probs = torch.softmax(logits, dim=0).tolist()
            perm_text_options = [perm["metadata"]["options"][c] for c in self.choices]
            for t, p in zip(perm_text_options, probs):
                aggregate_probs.setdefault(t, []).append(p)

        marginalized = {text_answer_2_letter[t]: torch.tensor(v).mean().item() for t, v in aggregate_probs.items()}
        return marginalized, aggregate_probs

    def _reduce_permutations(self, dataset_wpred):
        to_agg = {}
        for output in dataset_wpred:
            to_agg.setdefault(output["metadata"]["uid"], []).append(output)

        output_dataset_wpred = []
        for _, perms in to_agg.items():
            original_instance = copy.deepcopy(self._get_original_instance(perms))
            scores, all_scores = self._marginalize_across_permutations(perms)
            del original_instance["choice_logits"]
            original_instance["choice_probs"] = scores
            original_instance["generation"] = max(scores.items(), key=lambda x: x[1])[0]
            original_instance["choice_probs"] = scores
            original_instance["all_probs"] = all_scores
            original_instance["permutations"] = perms
            output_dataset_wpred.append(original_instance)
        return output_dataset_wpred

    def evaluation_postprocessing(self, metrics, dataset_with_predictions):
        dataset_with_predictions = self._reduce_permutations(dataset_with_predictions)
        metrics["debiased_accuracy"] = [
            float(d["generation"] == d["metadata"]["answer"]) for d in dataset_with_predictions
        ]
        return metrics, dataset_with_predictions
