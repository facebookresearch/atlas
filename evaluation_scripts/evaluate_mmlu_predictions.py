# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

SUBCATEGORIES = {
    "humanities": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
        "formal_logic",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "world_religions",
        "international_law",
        "jurisprudence",
        "professional_law",
    ],
    "Soc Sci.": [
        "high_school_government_and_politics",
        "public_relations",
        "security_studies",
        "us_foreign_policy",
        "human_sexuality",
        "sociology",
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_geography",
        "high_school_psychology",
        "professional_psychology",
    ],
    "STEM": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
        "college_chemistry",
        "high_school_chemistry",
        "college_biology",
        "high_school_biology",
        "college_computer_science",
        "computer_security",
        "high_school_computer_science",
        "machine_learning",
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
        "electrical_engineering",
    ],
    "other": [
        "global_facts",
        "miscellaneous",
        "professional_accounting",
        "business_ethics",
        "management",
        "marketing",
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
    "all": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ],
}


def load_predictions_file(file):
    predictions = {}
    for line in open(file):
        dp = json.loads(line)
        if "permuatations" in dp:
            dp["permutations"] = dp["permuatations"]
        original = [p for p in dp["permutations"] if p["metadata"]["is_original"]][0]
        dataset = original["metadata"]["dataset"].replace("_test", "").replace("_valid", "")
        uuid = original["metadata"]["question"] + str(original["metadata"]["options"])
        original_prediction = max(original["choice_logits"].items(), key=lambda x: x[1])[0]
        debiased_prediction = dp["generation"]
        predictions.setdefault(dataset, {})[uuid] = {
            "prediction": original_prediction,
            "debiased_prediction": debiased_prediction,
        }

    return predictions


def load_predictions(path, step=None, split=None):
    if os.path.isdir(path):
        predictions = {}
        for domain in os.listdir(path):
            predictions_path = os.path.join(path, domain, f"{domain}.{split}-step-{step}.jsonl")
            if not os.path.exists(predictions_path):
                raise ValueError(f"{predictions_path} expected but missing")
            predictions.update(load_predictions_file(predictions_path))
    else:
        predictions = load_predictions_file(path)
    return predictions


def load_gold_file(file):
    gold = {}
    for line in open(file):
        dp = json.loads(line)
        dataset = dp["dataset"].replace("_test", "").replace("_valid", "")
        uuid = dp["question"] + str(dp["options"])
        gold_answer = dp["answer"]
        gold.setdefault(dataset, {})[uuid] = gold_answer
    return gold


def score_categories(gold_answers, predictions, categories):
    acc = []
    debiased_acc = []
    for cat in categories:
        preds = predictions[cat]
        golds = gold_answers[cat]
        for question in golds.keys():
            pred = preds[question]
            gold = golds[question]
            acc.append(pred["prediction"] == gold)
            debiased_acc.append(pred["debiased_prediction"] == gold)
    acc = sum(acc) / len(acc)
    debiased_acc = sum(debiased_acc) / len(debiased_acc)
    return acc, debiased_acc


def main(predictions_file, gold_file, step=None, split=None):
    print(f"predictions for {predictions_file}")
    print(f"{'category': >15}\t{'Acc(%)':>15}\t{'Debias Acc(%)':>15}")
    predictions = load_predictions(predictions_file, step, split)
    gold_answers = load_gold_file(gold_file)
    print("-" * 47)
    for category_name, categories in SUBCATEGORIES.items():
        scores, debiased_scores = score_categories(gold_answers, predictions, categories)
        sc, db = f"{100*scores:0.2f}", f"{100*debiased_scores:0.2f}"
        print(f"{category_name: >15}\t{sc:>15}\t{db:>15}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Path to the written predictions file",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        help="Path to the written predictions file (zero-shot, 5-shot multi, full) or directory containing models (5-shot)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=16,
        help="only for 5-shot, specify the step to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="only for 5-shot, specify the split to evaluate",
    )
    args = parser.parse_args()
    main(args.predictions_path, args.gold_path, step=args.step, split=args.split)
