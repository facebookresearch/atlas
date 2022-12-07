# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of the experiment - also used as directory name "
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoint/",
            help="models are saved here",
        )
        self.parser.add_argument(
            "--model_path",
            type=str,
            default="none",
            help="Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)",
        )
        self.parser.add_argument(
            "--per_gpu_batch_size",
            default=1,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )

        self.parser.add_argument(
            "--per_gpu_embedder_batch_size",
            default=512,
            type=int,
            help="Embedder's batch size per GPU.",
        )

        self.parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank",
        )
        self.parser.add_argument(
            "--main_port",
            type=int,
            default=-1,
            help="Main port (for multi-node jobs)",
        )
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        self.parser.add_argument(
            "--log_freq",
            type=int,
            default=100,
            help="log train stats <log_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=5000,
            help="save model every <save_freq> steps during training",
        )
        self.parser.add_argument(
            "--train_data", nargs="+", default=[], help="list of space-separated paths to jsonl-formatted train sets"
        )
        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="list of space-separated paths to jsonl-formatted evaluation sets",
        )
        self.parser.add_argument("--write_results", action="store_true", help="save evaluation results to file")
        self.parser.add_argument(
            "--dont_write_passages",
            action="store_true",
            help="if writing results, passages can take up a lot of space, pass this flag not to write passages as part of dumped results",
        )

    def add_optim_options(self):
        self.parser.add_argument("--warmup_steps", type=int, default=1000, help="number of learning rate warmup steps")
        self.parser.add_argument("--total_steps", type=int, default=1000, help="total number of training steps")
        self.parser.add_argument(
            "--scheduler_steps",
            type=int,
            default=None,
            help="total number of step for the scheduler, if None then scheduler_total_step = total_step",
        )
        self.parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_retriever", type=float, default=1e-5, help="learning rate for retriever")
        self.parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            choices=["linear", "cosine", "fixed"],
            help="learning rate schedule to use",
        )
        self.parser.add_argument(
            "--weight_decay", type=float, default=0.1, help="amount of weight decay to apply in training"
        )
        self.parser.add_argument(
            "--save_optimizer", action="store_true", help="Pass flag to save optimizer state in saved checkpoints"
        )
        self.parser.add_argument("--epsilon", type=float, default=1e-6, help="adamw epsilon value")
        self.parser.add_argument("--alpha", type=float, default=1.0, help="adamw alpha value")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="adamw beta2 value")
        self.parser.add_argument(
            "--refresh_index",
            type=str,
            default="-1",
            help="index refresh schedule. format: startstep-endstep:refreshrate,startstep-endstep:refreshrate "
            "e.g. --refresh_index 0-100:10,100-1000000:500 will refresh the index every 10 steps for the first 100 steps, "
            "and then every 500 steps from step 100 to 1M."
            "Syntactic Sugar for a fixed schedule: can just pass in a single number e.g. --refresh_index 100 will refresh the index every 100 steps. "
            "-1 to never refresh.",
        )
        self.parser.add_argument("--shuffle", action="store_true", help="shuffle data for training")

        # memory optimizations:
        self.parser.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp16", "fp32", "bf16"],
            help="numerical precision - recommend bf16 if available, fp16 likely to be unstable for training",
        )
        self.parser.add_argument(
            "--shard_optim",
            action="store_true",
            help="train-time memory optimization: shards optimizer state over available GPUs using sharded data parallel, recommended for larger models",
        )
        self.parser.add_argument(
            "--shard_grads",
            action="store_true",
            help="train-time memory optimization: shards gradients over available GPUs using sharded data parallel, recommended for larger models",
        )
        self.parser.add_argument(
            "--use_gradient_checkpoint_reader",
            action="store_true",
            help="use gradient checkpointing in the reader",
        )
        self.parser.add_argument(
            "--use_gradient_checkpoint_retriever",
            action="store_true",
            help="use gradient checkpointing for retriever",
        )

    def add_modeling_options(self):
        self.parser.add_argument(
            "--reader_model_type",
            required=True,
            type=str,
            help="t5 Architecture for reader FID model, e.g. google/t5-xl-lm-adapt",
            choices=[
                "t5-small",
                "t5-base",
                "t5-large",
                "t5-3b",
                "t5-11b",
                "google/t5-v1_1-base",
                "google/t5-v1_1-large",
                "google/t5-v1_1-xl",
                "google/t5-v1_1-xxl",
                "google/t5-base-lm-adapt",
                "google/t5-large-lm-adapt",
                "google/t5-xl-lm-adapt",
                "google/t5-xxl-lm-adapt",
            ],
        )
        self.parser.add_argument(
            "--text_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--target_maxlength",
            type=int,
            default=None,
            help="Maximum length of target outputs in tokens when training the model. Targets longer than this will be truncated. No truncation if -1",
        )
        self.parser.add_argument("--n_context", type=int, default=1, help="number of top k passages to pass to reader")

        # Retriever modelling options
        self.parser.add_argument(
            "--passages",
            nargs="+",
            help="list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path",
        )
        self.parser.add_argument(
            "--max_passages",
            type=int,
            default=-1,
            help="maximum number of passages to index. -1 to read all passages in passage files",
        )
        self.parser.add_argument(
            "--retriever_model_path",
            type=str,
            default="facebook/contriever",
            help="path to contriever model to init from (overridden if passing a value to --model_path ",
        )
        self.parser.add_argument(
            "--retrieve_only",
            action="store_true",
            help="Pass this to prevent loading a reader, and only run retrieval evaluation",
        )
        self.parser.add_argument(
            "--train_retriever", action="store_true", help="Pass to train retriever as well as reader"
        )
        self.parser.add_argument(
            "--use_file_passages",
            action="store_true",
            help='uses passages in "passages" field in train or eval jsonl files rather than retrieving passages',
        )
        self.parser.add_argument(
            "--retriever_n_context",
            type=int,
            default=5,
            help="number of top k passages to use to train the retriever with",
        )
        self.parser.add_argument(
            "--gold_score_mode",
            type=str,
            choices=["evalnormsum", "loop", "ppmean", "emdr", "pdist", "adist"],
            default="ppmean",
            help="retriever training method. `pdist` is the name used in the paper for `ppmean`. `adist` is the name used in the paper for `evalnormsum`",
        )
        self.parser.add_argument(
            "--closed_book",
            action="store_true",
            help="Dont use retrieval - reduces to T5. Overrides n_context, n_context_retriever and encoder_format if they are set",
        )
        self.parser.add_argument(
            "--temperature_score", type=float, default=0.01, help="softmax temperature for retriever"
        )
        self.parser.add_argument(
            "--temperature_gold",
            type=float,
            default=0.01,
            help="softmax temperature for target distribution for retriever distillation",
        )
        self.parser.add_argument("--compute_crossattention_stats", action="store_true")
        self.parser.add_argument(
            "--filtering_overretrieve_ratio",
            type=int,
            default=2,
            help="if filtering, over-retrieve the topK by this factor, and then filter out undesirable results. Useful, Set to 1 only if using a task that doesn't filter retrieved results",
        )
        self.parser.add_argument("--freeze_retriever_steps", type=int, default=-1, help="freezes retriever for n steps")
        self.parser.add_argument(
            "--query_side_retriever_training",
            action="store_true",
            help="pass to enable query-side finetuning of retriever (unties the parameters of the contriever encoder's passage and query encoders, and freezes the passage encoder. Useful to avoid index refreshes.",
        )
        self.parser.add_argument(
            "--retrieve_with_rerank",
            action="store_true",
            help="pass this to enable reranking with fresh passage encoder for retriever",
        )
        self.parser.add_argument(
            "--n_to_rerank_with_retrieve_with_rerank",
            type=int,
            default=128,
            help="n passages to rerank when passing --retrieve_with_rerank. Higher is slower but more accurate. Recommend 64-128",
        )

        # input and output formatting options:
        self.parser.add_argument(
            "--decoder_format",  # TODO: decide whether to remove functionality
            type=str,
            default=None,
            help="format for decoder, model will be train on the format and evaluation will be performed with the format contrary to the decoder_prompt_format option",
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--decoder_prompt_format",
            type=str,
            default=None,
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(
            "--encoder_format",
            type=str,
            default="{query} title: {title} context: {text}",
            help="format string for reader's encoder preprocessing",
        )
        self.parser.add_argument(
            "--retriever_format",
            type=str,
            default="{title} {text}",
            help="format string for retriever's encoder preprocessing",
        )

        # Generation options
        self.parser.add_argument("--generation_max_length", type=int, default=128)
        self.parser.add_argument("--generation_min_length", type=int, default=None)
        self.parser.add_argument("--generation_length_penalty", type=float, default=1.0)
        self.parser.add_argument("--generation_num_beams", type=int, default=1)

        # Task-specific options:
        self.parser.add_argument(
            "--task",
            type=str,
            default=None,
            choices=["base", "mlm", "lm", "multiple_choice", "kilt", "section", "fever", "qa"],
            help="Task performed by the model. Used to setup preprocessing, retrieval filtering, evaluations, etc.",
        )

        # MLM task options:
        self.parser.add_argument(
            "--mlm_noise_density",
            type=float,
            default=0.15,
            help="how much of an input text should be masked by masking spans ",
        )
        self.parser.add_argument(
            "--mlm_mean_noise_span_length", type=float, default=3, help="average length of an MLM masking span"
        )
        self.parser.add_argument(
            "--min_words_per_lm_instance",
            type=int,
            default=None,
            help="Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section Generation",
        )

        # LM task options:
        self.parser.add_argument(
            "--min_lm_context_ratio",
            type=float,
            default=0.5,
            help="Splits text into two segments for language modelling.'\
                'Left segment is conditioning context, right segment is for generating.'\
                'The left segment must be more than min_lm_context_ratio of the the right segment",
        )
        self.parser.add_argument(
            "--max_lm_context_ratio",
            type=float,
            default=0.5,
            help="Splits text into two segments for language modelling.'\
                'Left segment is conditioning context, right segment is for generating.'\
                'The left segment must be less than than max_lm_context_ratio of the the right segment",
        )

        # Open-domain task options:
        self.parser.add_argument(
            "--qa_prompt_format",
            type=str,
            default="question: {question} answer: <extra_id_0>",
            help="How to format question as input prompts when using --task qa",
        )

        # Multiple Choice task options:
        self.parser.add_argument(
            "--multiple_choice_num_options",
            type=int,
            default=4,
            help="How many choice options for multiple choice QA (MMLU is 4)",
        )
        self.parser.add_argument(
            "--multiple_choice_train_permutations",
            choices=["single", "cyclic", "all"],
            default="single",
            type=str,
            help="Whether to train with answer order permutations When training on multiple choice (e.g. MMLU)."
            " Can improve results by de-biasing models's preferences for arbitrary answer orderings. Recommend training with 'all'. "
            "single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations'",
        )
        self.parser.add_argument(
            "--multiple_choice_eval_permutations",
            choices=["single", "cyclic", "all"],
            default="single",
            type=str,
            help="Whether to evaluate with answer order permutations for multiple choice (e.g. MMLU)."
            " Can improve results by de-biasing models's preferences for arbitrary answer orderings. Best results with 'all' but very slow. 'cyclic' is a good compromise. "
            "single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations'",
        )

    def add_index_options(self):
        self.parser.add_argument(
            "--load_index_path",
            default=None,
            type=str,
            help="path for loading the index, passage embeddings and passages",
        )
        self.parser.add_argument(
            "--save_index_path",
            default=None,
            type=str,
            help="path for saving the index and/or embeddings",
        )
        self.parser.add_argument(
            "--save_index_n_shards",
            default=128,
            type=int,
            help="how many shards to save an index to file with. Must be an integer multiple of the number of workers.",
        )
        self.parser.add_argument(
            "--index_mode",
            type=str,
            default="flat",
            help="Use flat torch index or a faiss index for retrieving the k nearest neighbors",
            choices=["flat", "faiss"],
        )
        # faiss options:
        self.parser.add_argument(
            "--faiss_index_type",
            type=str,
            default="flat",
            help="IVFFlat, IndexFlatIP, IVFScalarQuantizer or IndexIVFPQ with faiss-gpu",
            choices=["ivfflat", "flat", "ivfsq", "ivfpq", "pq"],
        )
        self.parser.add_argument("--faiss_code_size", type=int, default=None, help="Parameter for PQ/SQ quantization")

    def print_options(self, opt):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f"\t(default: {default_value})"
            message += f"{k:>30}: {str(v):<40}{comment}\n"

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        with open(expr_dir / "opt.log", "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        if opt.closed_book:  # override flags to enable closed book mode
            opt.n_context = 1
            opt.retriever_n_context = 1
            opt.encoder_format = "{query}"
            opt.use_file_passages = True
        if opt.gold_score_mode == "pdist":  # allow paper name of retriever losses
            opt.gold_score_mode = "ppmean"
        if opt.gold_score_mode == "adist":  # allow paper name of retriever losses
            opt.gold_score_mode = "evalnormsum"
        if (
            opt.use_file_passages
        ):  # if passing use_file_passges, the following should be false (There is no retreiver loaded in this case)
            opt.train_retriever = False
            opt.query_side_retriever_training = False
            opt.use_gradient_checkpoint_retriever = False
        return opt


def get_options():
    options = Options()
    options.add_index_options()
    options.add_modeling_options()
    options.add_optim_options()
    return options
