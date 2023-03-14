# Atlas: Few-shot Learning with Retrieval Augmented Language Models

This repository contains pre-trained models, corpora, indices, and code for pre-training, finetuning, retrieving and evaluating for the paper [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/pdf/2208.03299.pdf)

We jointly pretrain a retrieval-augmented seq2seq language model, comprised of a passage-based dense retriever and a encoder-decoder language model. 
We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and
study the impact of the content of the document index, showing that it can easily be updated.
Notably, Atlas reaches over 45% accuracy on Natural Questions using only 64 examples when supplied with wikipedia index from 2018,
outperforming a 540B parameters model by 6% despite having 50x fewer parameters.
Atlas also works very well when finetuned on larger datasets - when finetuned on the full Natural Questions data, Atlas sets a new state-of-the-art of 64%, 8 points higher than the current state of the art.

This repository supports pretraining and finetuning, for *both* large and small datasets. This repository can be supports the following features:
* Training large fusion-in-decoder seq2seq models, tested up to 11B parameters
* Distilling relevance signals from fusion-in-decoder models into dense retrieval models using a variety of different distillation approaches.
* Performing end-to-end retrieval-augmented training over a user-supplied corpus of passages (tested with up to 400M passages, ~40B words) with retrieval-in-the-training-loop
* Support for training on Masked-Language modelling, prefix-language modelling, wikipedia section generation, Open-Domain Question Answering, Multiple Choice Question Answering, Fact checking, and KILT (arbitrary seq2seq tasks can also be supported)
* A fast, parallel distributed GPU-based exact and approximate maximum inner product search for dense vector retrieval
* Support for fast in-place index refreshes
* Various memory optimizations and methods for maintaining fast and accurate retrieval while training retrievers in-the-loop.
* plus more, see the command line arguments or the readme for additional features

## Table of Contents

* [Installation](#installation)
* [Getting Started and Codebase at a Glance](#getting-started-and-codebase-at-a-glance)
* [Available Data and Models for download](#available-data-and-Models-for-download)
  * [Corpora](#corpora)
  * [Models](#models)
  * [Pre-built Indices](#prebuilt-indices)
* [Tasks](#tasks)
  * [Basic](#base-task)
  * [Masked Language Modelling](#mlm-task)
  * [Wikipedia Section Generation](#section-task)
  * [Open-Domain Question Answering (e.g. NaturalQuestions, TriviaQA, TempLama)](#qa-task)
  * [Multiple Choice Question Answering (e.g. MMLU)](#mcqa-task)
  * [Fact Checking](#fever-task)
  * [KILT](#kilt-task)
* [Retrieval and Index Details](#retrieval-and-index-details)
  * [Flat vs Faiss](#flat-vs-faiss)
  * [Index Saving and Loading](#index-saving-and-loading)
  * [Strategies for dealing with stale indices](#strategies-for-dealing-with-stale-indices)
    * [Index Refresh](#strategies-for-dealing-with-stale-indices)
    * [Over-Retrieve with Reranking](#strategies-for-dealing-with-stale-indices)
    * [Query-Side Finetuning](#strategies-for-dealing-with-stale-indices)
  * [Retrieve-only mode](#retrieve-only-mode)
  * [Using pre-retrieved or cached passages](#using-pre-retrieved-or-cached-passages)
* [Other features](#other-features)
  * [Closed book mode](#closed-book-mode)
  * [Specifying formats](#specifying-formats)
  * [Implementing your own task](#implementing-your-own-task)
* [Full list of command line flags](#full-list-of-command-line-flags)
* [Citing](#citing)
* [LICENSE](#license)
  * [Code License:](#code-license)
  * [Data License:](#data-license)


## Installation

The Atlas codebase uses the following dependencies:

* python 3 (tested with 3.8)
* fairscale (tested with 0.4.6)
* transformers (tested with 4.18.0)
* numpy (tested with 1.22.4)
* faiss (tested with 1.7.2)

We recommend installing using conda. The following will install all dependencies:
```
git clone https://github.com/facebookresearch/atlas.git
cd atlas
conda create --name atlas-env python=3.8
conda activate atlas-env
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3
pip install -r requirements.txt
```

## Getting Started and Codebase at a Glance

The Atlas repository provides functionality for training and evaluating retrieval-augmented generation models, comprised of an encoder-decoder language model, and dense-vector retriever.
<!-- We current  functionality for *jointly* training an encoder-decoder language model and retrieval-augmented with a dense retriever. -->
We currently support T5 architectures for the encoder-decoder language model and Contriever architectures for the retriever (Support for other architectures is not currently planned, but PRs are welcome).
Atlas models are comprised of a Contriever retriever and fusion-in-decoder (FID) architecture (which uses T5). You can learn more about the Contriever and FID [here](https://github.com/facebookresearch/FiD) and [here](https://github.com/facebookresearch/contriever) respectively if desired, but all required functionality has been reimplemented in this codebase.

The biggest difference to most standard NLP training codebases is that Atlas performs retrieval on-the-fly, and can refresh its retrieval embeddings index in-place.
This is achieved using a custom-designed distributed GPU index, which automatically handles fast and scale-able retrieval.
 
**A note on how retrieval is accomplished:** 
When launching a training or evaluation run, the codebase will first load pretrained models, then each GPU worker will load a shard of the supplied passages to retrieve from -- if there are N GPUs, each will load a shard of 1/N passages. 
Each worker will then embed its shard of the passages using the retriever embedder, and keep the passage embedding shard in GPU memory (and optionally build a FAISS index).
At this point, the passage and embedding shards (referred to as "the index") can be optionally saved to disk to avoid the need to recompute indices for every run. 
Retrieval is performed in parallel, with each GPU worker performing an exact maximum inner product search for all the queries for its shard. 
More details on retrieval are given in the [Retrieval and Index Details](#retrieval-and-index-details) section.
*Note that all of the above is all handled automatically by the codebase*, so users should not need to know or worry too much about how embedding, index refresh or retrieval is accomplished, other than 
1) noting that they can easily retrieve from any set of passages that they like by just passing in paths to suitably-formatted passages on disk (or any saved index) 
2) noting that embedding, index refresh retrieving will get faster with more GPU workers.
3) Depending on how many GPUs and CPU memory is available, Atlas can support training models with 11B+ parameters and indices of 400M+ vectors, or ~40 billion words (assuming ~100 words a passage)

Training and Evaluation uses a data-parallel model: for N GPU workers, each processes 1/N of the total mini-batch of data. To save memory at training time, optimizer state and gradients can be sharded using fairscale's ShardedDataParallel. 

All data files (retriever passages and train/dev/test data) should be supplied in the form of [jsonlines](https://jsonlines.org/) ("jsonl") files.
Passages to retrieve from should consist of json-serialized objects with `text` and `title` text fields, one passage per line.
Example passage files are available for wikipedia (see [corpora](#corpora)).
Train/dev/test data files should be json-serialized objects, one instance per line. The name of the fields is task dependent (covered in detail in [Tasks](#tasks)), but e.g. for NaturalQuestions, the required fields are `question` (a question string) and `answers` (a list of reference answer strings)

The codebase has two entrypoint scripts: `train.py` for training, and `evaluate.py` for test-time evaluation (and [stand-alone retrieval](#retriever-only-mode), if you want).
You can list the full Atlas functionality by printing the command-line flags using `python train.py -h` (full output [here](#full-list-of-command-line-flags))

*The easiest way to illustrate the codebase is with an example:*

The following example shows an example use case: few-shot finetuning and evaluating on NaturalQuestions with Atlas-large (which are also available as a runnable sbatch scripts in `example_scripts/nq/`), retrieving from a wikipedia dump from 2018 (of about 30M passages)

```bash
# assumes 4 nodes, each with 8 GPUs
DATA_DIR=./atlas_data
SIZE=large # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)

# download the NQ data
python preprocessing/prepare_qa.py --output_directory ${DATA_DIR}/data/
# download the Wikipedia 2018 corpus
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR} 
# downloads pretrained Atlas-large
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}  

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/nq_data/train.64-shot.jsonl"
EVAL_FILES="${DATA_DIR}/data/nq_data/dev.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=my-nq-64-shot-example
TRAIN_STEPS=30

srun python train.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode pdist \ # loss function for retriever (see paper)
    --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever\ # save GPU memory with gradient checkpointing at expense of speed
    --precision fp32 \ # use "bf16" if supported by your GPUs, fp16 is usually unstable
    --shard_optim --shard_grads \ # Save GPU memory using these optimizations
    --temperature_gold 0.01 --temperature_score 0.01 \ 
    --refresh_index -1 \ # for fewshot finetune, refreshing the index (i.e. recomputing the embeddings) is expensive and not really worth it
    --query_side_retriever_training\ # instead, for fewshot runs, finetuning only the query-encoder of Contriever works well. Remove this flag to finetune whole retriever
    --target_maxlength 16 \ # max length of generation
    --reader_model_type google/t5-${SIZE}-lm-adapt \ # architecture of Atlas
    --dropout 0.1 --weight_decay 0.01 --lr 4e-5 --lr_retriever 4e-5 --scheduler linear \ # optimization flags
    --text_maxlength 512 \ # max length of question + passage when concatenated
    --model_path "${DATA_DIR}/models/atlas/${SIZE}" \ # path to the pretrained Atlas model we just downloaded (pass 'none' to init from plain t5 and Contriever)
    --train_data "${DATA_DIR}/data/nq_data/train.64-shot.jsonl" \ # path the 64-shot train dataset we just downloaded 
    --eval_data "${DATA_DIR}/data/nq_data/dev.jsonl" \ # path the NQ dev dataset we just downloaded, to evaluate on when training is done
    --per_gpu_batch_size 1 \
    --n_context 40 \ # pass the top 40 passages from the retriever to the language model
    --retriever_n_context 40 \ # finetune the retriever with the top 40 passages
    --name ${EXPERIMENT_NAME} \ # name of experiment (also the name of the directory the logs and models will be saved to) 
    --checkpoint_dir ${SAVE_DIR} \ # logs and model checkpoints will be saved to ${SAVE_DIR}/${EXPERIMENT_NAME}
    --eval_freq ${TRAIN_STEPS} \ # eval after we finish training
    --log_freq 4 \ # log stats every 4 training steps. Logs will write to ${SAVE_DIR}/${EXPERIMENT_NAME}/run.log but will also write tensorboard logs if installed
    --total_steps ${TRAIN_STEPS} \ # train for this many steps
    --warmup_steps 5 \
    --save_freq ${TRAIN_STEPS} \ # for this example, we'll save one checkpoint, after the training is complete
    --main_port $port \ # for distributed training
    --write_results \ # write predictions - they will get saved in the checkpoint folder, ${SAVE_DIR}/${EXPERIMENT_NAME}
    --task qa \ # we're doing the QA task
    --index_mode flat \ # don't use faiss, keep index flat (recommended unless using very large indices or very constrained on GPU memory)
    --passages "${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl" "${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"\ # pass in the wikipedia passages to index and retrieve from (we use both the text and infoboxes)
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index # save the index we built to this path
```

The training script will first embed an index for wikipedia 2018, and then save it under the checkpoint folder (`${SAVE_DIR}/${EXPERIMENT_NAME}`). 
The training script will then fewshot-finetune an Atlas-large NQ model for 30 steps, retrieving from all of wikipedia 2018. 
This particular script finetunes the query encoder of the retriever and the FID, whilst keeping the passage encoder frozen (see the paper, or [below](#strategies-for-dealing-with-stale-indices) for further details).
Th script will then evaluate on the dev set and save the checkpoint. 
You can inspect the experiment logs at `${SAVE_DIR}/${EXPERIMENT_NAME}/run.log` and observe a NQ-dev Exact match score of ~38 has been logged (our run was 38.4), and written predictions which can be inspected.

To evaluate the model, (e.g. on heldout test data) we can use the `evaluate.py` entrypoint script:

```bash
srun python evaluate.py \
    --name 'my-nq-64-shot-example-evaluation' \
    --generation_max_length 16 \
    --gold_score_mode "pdist" \
    --precision fp32 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/step-30 \ #now, we point this to the model we just trained
    --eval_data "${DATA_DIR}/data/nq_data/dev.jsonl ${DATA_DIR}/data/nq_data/test.jsonl" \ # lets evaluate on the dev data and the test data this time
    --per_gpu_batch_size 1 \
    --n_context 40 --retriever_n_context 40 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --load_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index\ # rather than re-embed all the wikipedia passages again, lets load them from the index we just saved above
    --write_results # write the inference results
```
This script will load the model, and since we specified to load a saved index via `--load_index_path`, it will load an index rather than embed from passages as before. 
It will then evaluate the development and test sets.
Inspecting the saved logs at `${SAVE_DIR}/my-nq-64-shot-example-evaluation/run.log`, we will see the same exact match score for the dev set that we got before, and a test score of ~38 (in our case 38.8 EM).

The rest of this readme describes data, code and functionality in detail.

## Available Data and Models for download

Atlas's wikipedia corpora, the pretrained models and pre-built wikipedia indices are available for download at this time. 

Click to expand:
<details>
<summary>
<h4 name="corpora">Corpora</h4>
</summary>

The preprocessed wikipedia dumps we use for retrieving and pretraining Atlas can be downloaded as follows:

```bash
python preprocessing/download_corpus.py --corpus {corpus download key} --output_directory ${DATA_DIR} 
```
The above string will download a corpus and unzip it to `${DATA_DIR}/{corpus download key}` 

The available corpora are given below:

| Corpus Name      | Corpus Download Key | Description | Size |
| ----------- | ----------- | --------|  ---- |
| enwiki-dec2017      | `corpora/wiki/enwiki-dec2017` | Wikipedia dump from Dec 2017, preprocessed into passages       |  30.4M (26.9M text, 2.7M  infobox)| 
| enwiki-dec2018      | `corpora/wiki/enwiki-dec2018` | Wikipedia dump from Dec 2018, preprocessed into passages (recommended for NQ, TriviaQA) | 32.1M (28.4M text, 3.7M infobox) |
| enwiki-aug2019      | `corpora/wiki/enwiki-aug2019` |  Wikipedia dump from August 2019, preprocessed into passages       | 33.1M (29.4M text, 3.8M infobox)  |
| enwiki-dec2020      | `corpora/wiki/enwiki-dec2020` |  Wikipedia dump from Dec 2020, preprocessed into passages       | 35.6M (31.5M text, 4.1M infobox) |
| enwiki-dec2021      | `corpora/wiki/enwiki-dec2021` | Wikipedia dump from Dec 2021, preprocessed into passages       | 37.5M (33.1M text, 4.3M infobox) |

Passage files are jsonl formatted, with one passage serialized as a json object per line. By default, each passage should be formatted as follows:

```python
{
    "id": "0", # passages should have a unique id
    "title": "Orchid", # should specify the title of the page passage comes from (can be empty string if there's no good title)
    "text": "Orchids are easily distinguished from other plants, as they share some very evident derived characteristics or synapomorphies. Among these are: bilateral symmetry of the flower (zygomorphism), many resupinate flowers, a nearly always highly modified petal (labellum), fused stamens and carpels, and extremely small seeds.", # main text of passage
    "section": "Description" # Optional, section title, if non empty this field is appended to the title as {title}: {section} by default
    ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```

Creating your own passage files to use with Atlas should be straightforward if you follow the above formatting.

We cannot open-source the common-crawl indices used in the paper at this time.
</details>

<details>
<summary>
<h4 name="models">Models</h4>
</summary>

We are open-sourcing pretrained Atlas models at base, large, xl and xxl sizes. These include both the pretrained retriever and reader weights.
In addition, we're open-sourcing our strongest-performing fully-finetuned NaturalQuestions Atlas models, for users who want to perform state-of-the-art QA inference (or finetune them on other QA tasks).
Models can be downloaded as follows:

```bash
python preprocessing/download_model.py --model {model download key} --output_directory ${DATA_DIR} 
```

This will download the requested model to `${DATA_DIR}/{model download key}`, and it can then be used in scripts by passing `${DATA_DIR}/{model download key}` to `--model_path`.
The following table details the available models:

| Model | Model Download Key | Description | Parameters (reader / retriever) |
| ----------- | ----------- | --------| ----|
| Atlas-xxl | `models/atlas/xxl` | Pretrained Atlas XXL model | 11B / 110M |
| Atlas-xl | `models/atlas/xl` | Pretrained Atlas XL model | 3B / 110M |
| Atlas-large | `models/atlas/large` | Pretrained Atlas large model | 770M / 110M |
| Atlas-base | `models/atlas/base` | Pretrained Atlas base model | 220M / 110M |
| NQ-finetuned Atlas-xxl | `models/atlas_nq/xxl` |Atlas XXL model, finetuned on Natural Question | 11B / 110M |
| NQ-finetuned Atlas-xl | `models/atlas_nq/xl` | Atlas XL model, finetuned on Natural Question | 3B / 110M |
| NQ-finetuned Atlas-large | `models/atlas_nq/large` | Atlas large model, finetuned on Natural Question | 770M / 110M |
| NQ-finetuned Atlas-base | `models/atlas_nq/base` |Atlas base model, finetuned on Natural Question| 220M / 110M |
</details>

<details>
<summary>
<h4 name="prebuilt-indices">Pre-built Indices</h4>
</summary>

Atlas will automatically build an index if none is provided. This is convenient, but can take a long time, especially with fewer GPU workers, or if the index is very large.

We have therefore made precomputed indices available for download for the wiki-dec2018 corpus for the pretrained Atlas checkpoints, and for nq-finetuned Atlas checkpoints

These can be downloaded as follows :
```bash
python preprocessing/download_index.py --index {index download key} --output_directory ${DATA_DIR} 
```

The above script will download the requested pretrained index and save them to `${DATA_DIR}/{index download key}`. 
They can then be used in training or evaluation by passing them to `--load_index_path`. 
More details on index saving and loading are given in [Retrieval and Index Details](#retrieval-and-index-details). 
The following indices are available for download:

| Index  | Index Download Key | Corresponding Model |  Description |
| --------| ------| --------| ------|
| Atlas XXL wiki-dec2018 index | `indices/atlas/wiki/xxl` | `models/atlas/xxl` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-xxl model |
| Atlas XL wiki-dec2018 index | `indices/atlas/wiki/xl` | `models/atlas/xl` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-xl model |
| Atlas large wiki-dec2018 index | `indices/atlas/wiki/large` | `models/atlas/large` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-large model |
| Atlas base wiki-dec2018 index | `indices/atlas/wiki/base` | `models/atlas/base` | Precomputed index for the wiki-dec2018 corpus for the pretrained Atlas-base model |
| Atlas-nq XXL wiki-dec2018 index | `indices/atlas_nq/wiki/xxl` | `models/atlas_nq/xxl` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas xxl model |
| Atlas-nq XL wiki-dec2018 index | `indices/atlas_nq/wiki/xl` | `models/atlas/xl` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas xl model |
| Atlas-nq large wiki-dec2018 index | `indices/atlas_nq/wiki/large` | `models/atlas/large` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas large model |
| Atlas-nq base wiki-dec2018 index | `indices/atlas_nq/wiki/base` | `models/atlas/base` | Precomputed index for the wiki-dec2018 corpus for the natural-questions-finetuned Atlas base model |
</details>

## Tasks

Atlas can train (or evaluate) on any supervised learning task which can be formulated in a "seq2seq" format, where there is a sequence of 1 or more tokens comprising an input *query* and a sequence of 1 or more tokens comprising an output *target*.
For example, a query might be a question, `Where is the Bermuda Triangle?`, and a target might be the answer to that question, `Western part of the North Atlantic Ocean`.
This way of modelling will be familiar to users of models like T5 or BART. Anywhere these models could be used, Atlas can be used too, using the exact same data: Atlas will learn to retrieve passages from its retrieval index by itself - annotations for associating passages to (`query`, `target`) pairs are not used.

The Atlas codebase configures what task it is doing, and what evaluation metrics to call using the `--task` command line argument. 
We have implemented a `base` task, with only the most basic support for seq2seq training, but provide more fully-featured functionality for Masked Language Modelling (`mlm`), Language Modelling (`lm`), Wikipedia section generation (`section`), Open-domain QA (`QA`), Multiple choice QA (`multiple_choice`), fact checking (`fever`), and the KILT suite (`kilt`), 
All tasks expect input data formatted as jsonl format, but the specific field names are task specific. Some tasks have additional command line args, and specialized evaluation.
Adding new tasks is straightforward, and described [here](#defining-your-own-task).

The tasks are described in more detail below, and most have example commands in `examples/{task}/` (click to expand).

<details>
<summary>
<h4 name="base-task">Base Task</h4>
</summary>


This is the most basic task available, and is probably not the best option for you, especially if your task closely resembles one the other implemented tasks.

Specify this task by passing `--task base` to either `train.py` or `evaluate.py` 

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py` or `evaluate.py` space-separated lists to `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
This task expects input files to have a `query` field with the input query string and a `target` field with the output query string, e.g.:

```json
{"query": "input to Atlas", "target": "desired generation from Atlas"}
```

The evaluation loop will calculate evaluation loss and the fraction of eval data examples where Atlas generates an output that exactly matches the target.
If you pass `--write_results` to the script, Atlas predictions on the eval data will be written to the save checkpoint directory with the following format:

```json
{"query": "input to Atlas", "answers": ["desired generation from Atlas"], "generation": "Atlas's prediction for the query", "passages": ["list of retrieved passages"]}
```

</details>

<details>
<summary>
<h4 name="mlm-task">Masked Language Modelling</h4>
</summary>

The Masked Language modelling task implements the Masked Language Modelling pretraining task as introduced by [T5](https://arxiv.org/abs/1910.10683).
This is the task we use to pretrain the main Atlas in the paper.

Specify this task by passing `--task mlm` to `train.py`.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
These files should be comprised of JSON objects with the following format:
```python
{
  "text": "text passage to apply noise to and train to de-noise",
  "id": "unique id of text passage"
  ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
The intention is that the same files that you use for the retrieval corpus, (passed to `--passages`) can be used as training data.
The task will apply the T5 noise function to `text` field, to automatically create inputs and target generations.

The MLM task will prevent Atlas from retrieving the passage that it is trying to de-noise. It does this by filtering out any passage from retrieved results which have same `id` field as the instance Atlas is de-noising. 
This functionality is important if the de-noising training data and the passages Atlas is retrieving from are the same corpus.

This task has the following task specific args:
```
  --mlm_noise_density MLM_NOISE_DENSITY
      how much of an input text should be masked by masking spans (default: 0.15)
  --mlm_mean_noise_span_length MLM_MEAN_NOISE_SPAN_LENGTH
      average length of an MLM masking span (default: 3)
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
      Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section generation (default: None)
```

If you pass `--write_results`, Atlas will write its mask-filling predictions to file.

Atlas will log the following evaluation metrics for MLM during its evaluation loop: 
* `eval_loss`: evaluation reader loss of generated mlm mask-fill spans
* `accuracy`: fraction of perfectly de-noised mask-fill spans
* `f1`: token f1 fraction of correct de-noised mask-fill spans
* `rouge_1`: rouge 1 score of generated mask-fill spans relative to the gold reference masked spans
* `rouge_2`: rouge 2 score of generated mask-fill spans relative to the gold reference masked spans
* `rouge_L`: rouge L score of generated mask-fill spans relative to the gold reference masked spans

</details>

<details>
<summary>
<h4 name="lm-task">Language Modelling</h4>
</summary>

Atlas can be trained to do Left-to-Right Language Modeling by passing `--task lm` to `train.py`.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
These files should be comprised of JSON objects with the following format:
```python
{
  "text": "text passage to train Atlas to generate",
  "id": "unique id of text passage"
  ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
The intention is that the same files that you use for the retrieval corpus, (passed to `--passages`) can be used as training data.
The task will preprocess the `text` field automatically, dividing it into two random segments - the left part serves as conditioning context, and the right part is the text the Atlas model will be trained to generate as a continuation.

The LM task will prevent Atlas from retrieving the same passage that it is trying to generate. It does this by filtering out any passage from retrieved results which have same `id` field as the instance Atlas is generating. 
This functionality is important if the de-noising training data and the passages Atlas is retrieving from are the same corpus.

This task has the following task specific args:
```
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
      Instances with fewer than min_words_per_lm_instance instances will be skipped for  MLM/LM/Section generation (default: None)
  --min_lm_context_ratio MIN_LM_CONTEXT_RATIO
      Splits text into two segments for language modelling.' 'Left segment is conditioning context, right segment is for generating.' 'The left segment must be more than min_lm_context_ratio of
      the right segment (default: 0.5)
  --max_lm_context_ratio MAX_LM_CONTEXT_RATIO
      Splits text into two segments for language modelling.' 'Left segment is conditioning context, right segment is for generating.' 'The left segment must be less than max_lm_context_ratio
      of the right segment (default: 0.5)
```

If you pass `--write_results`, Atlas will write its lm predictions to file.

Atlas will log the following evaluation metrics for LM during its evaluation loop: 
* `eval_loss`: evaluation reader loss of continuations for the reference data
* `accuracy`: fraction of perfectly predicted continuations
* `f1`: token f1 fraction of correct generated continuations
* `rouge_1`: rouge 1 score of generated continuations relative to the gold reference continuations
* `rouge_2`: rouge 2 score of generated continuations relative to the gold reference continuations
* `rouge_L`: rouge L score of generated continuations relative to the gold reference continuations

</details>

<details>
<summary>
<h4 name="section-task">Wikipedia Section Generation</h4>
</summary>

Atlas can be trained to generate the text of a wikipedia passage given its title and section title, by passing  `--task section` to `train.py`.

Train/validation/test data for this task should consist of jsonl files, which should have the form of the `text-list-100-sec.jsonl` files in the wikipedia dumps.
These can be obtained by following the instructions in [Available Data and Models for download](#available-data-and-Models-for-download), for example the training file: `enwiki-dec2018/text-list-100-sec.jsonl`.
These files should be comprised of JSON objects, one per line, with the following format:
```json
{
  "id": "3793043", 
  "title": "Bermuda Triangle",
  "section": "Compass variations",
  "text": " Compass problems are one of the cited phrases in many Triangle incidents. While some have theorized that unusual local magnetic anomalies may exist in the area, such anomalies have not been found. Compasses have natural magnetic variations in relation to the magnetic poles, a fact which navigators have known for centuries."
}
```
The task will automatically format the input query to the model as "{Title}, {Section}" - e.g. in this example, the input to Atlas will be constructed as `Bermuda Triangle, Compass Variations`. The output will be the `text` field of the example.
The `section` task will prevent Atlas from retrieving the same passage that it is trying to generate. It does this by filtering out any passage from retrieved results which have same `id` field as the instance Atlas is generating. 

This task has the following task specific args:
```
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
      Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section generation (default: None)
```
If you pass `--write_results`, Atlas will write its generated predictions for the text for Wikipedia sections to file.

Atlas will log the following evaluation metrics for `section` during its evaluation loop: 
* `eval_loss`: evaluation reader loss of continuations for the reference data
* `accuracy`: fraction of perfectly predicted continuations
* `f1`: token f1 fraction of correct generated continuations
* `rouge_1`: rouge 1 score of generated continuations relative to the gold reference continuations
* `rouge_2`: rouge 2 score of generated continuations relative to the gold reference continuations
* `rouge_L`: rouge L score of generated continuations relative to the gold reference continuations

</details>


<details>
<summary>
<h4 name="qa-task">Open-Domain Question Answering (e.g. NaturalQuestions, TriviaQA, TempLama)</h4>
</summary>

Atlas can be trained to answer open-domain QA questions by passing `--task qa` to `train.py` or `evaluate.py`.
There is a worked example of QA in the [Getting Started and Codebase at a Glance](#getting-started-and-codebase-at-a-glance) section.
We use this task for the NaturalQuestions, TriviaQA and TempLama datasets in the paper.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format:
```python
{
  "question": "where is the bermuda triangle",
  "answers": ["Western part of the North Atlantic Ocean"],
   ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
The question will be formatted according to the task specific argument `--qa_prompt_format`, which defaults to `question: {question} answer: <extra_id_0>`.
For example above, the question would be automatically formatted as input queries to Atlas as `question: where is the bermuda triangle answer: <extra_id_0>`.
The supervision target is obtained from the `target` field. If this field does not exist, the supervision target will get selected at random from the available answers in the `answers` field, and formatted as `<extra_id_0> {answer}`.

If you pass `--write_results`, Atlas will write its predicted answers to file.

Atlas will log the following evaluation metrics for open domain QA during its evaluation loop: 
* `eval_loss`: evaluation reader loss of evaluation answers.
* `exact_match`: Open-domain QA exact match score of generated answers
* `f1`: Open-domain QA F1 score of generated answers

#### Natural Questions & TriviaQA

You can download the NaturalQuestions and TriviaQA data by calling:

```bash
python preprocessing/prepare_qa.py --output_directory ${DATA_DIR} 
```

which will download `train.jsonl`, `train.64-shot.jsonl` (the fewshot training dataset we use), `dev.jsonl` and `test.jsonl` to `${DATA_DIR}/data/nq_data` and `${DATA_DIR}/data/triviaqa_data`.

Example scripts for running fewshot and standard finetuning and evaluation with a wikipedia index for NQ can be found in `examples/nq`. This script can be used for TriviaQA by swapping the train/dev/test files.

#### TempLama

We defined a cloze-question answering task for assessing index faithfulness and temporal transfer, derived from the TempLAMA dataset.

You can download the TempLAMA data and create and format our derived dataset by calling the following script:

```bash
python preprocessing/prepare_templama.py --output_directory ${DATA_DIR} 
```

which will create the files  `temp_lama.train.2017.jsonl`, `temp_lama.valid.2017.jsonl`, `temp_lama.test.2017.jsonl`, `temp_lama.train.2020.jsonl`, `temp_lama.valid.2020.jsonl`, `temp_lama.test.2020.jsonl` under `${DATA_DIR}/data/templama_data/`.
These files will contain cloze questions, with answers specific to that year. 

Example scripts for running training and evaluation for TempLama can be found at `examples/templama`. (note the use of `qa_prompt_format {question}`, which switches off the automatic QA prompt formatting used for TriviaQA and NQ)

</details>

<details>
<summary>
<h4 name="mcqa-task">Multiple Choice Question Answering (e.g. MMLU)</h4>
</summary>

Atlas can be trained to answer multiple choice questions by passing `--task multiple_choice` to `train.py` or `evaluate.py`.
We use this task for our experiments with MMLU.

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format:
```python
{
  "question": "Which of the following is the body cavity that contains the pituitary gland?", 
  "options": {
    "A": "Abdominal",
    "B": "Cranial",
    "C": "Pleural", 
    "D": "Spinal"
    ... # you can have more (or fewer) answer options as long as they have alphabetically consecutive upper case letter keys, starting at A
  }, 
  "answer": "B",
  ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
These will get automatically formatted into input queries for Atlas of the form `question: {question} answers: (A) {options['A']} (B) {options['B']} (C) {options['C']} (D) {options['D']} Answer: <extra_id_0>`, with target generations of the format `<extra_id_0> {answer letter}`.
The example above would get formatted to: `question: {Which of the following is the body cavity that contains the pituitary gland? answers: (A) Abdominal (B) Cranial (C) Pleural (D) Spinal Answer: <extra_id_0>`, with the target generation `{extra_id_0} B`.


Multiple-Choice QA has the following task specific args:
```
  --multiple_choice_num_options
      How many choice options for multiple choice QA (MMLU is 4) (default: 4)
  --multiple_choice_train_permutations {single,cyclic,all}
      Whether to train with answer order permutations When training on multiple choice (e.g. MMLU). Can improve results by de-biasing models's preferences for arbitrary answer orderings. Recommend
      training with 'all'. single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations' (default: single)
  --multiple_choice_eval_permutations {single,cyclic,all}
      Whether to evaluate with answer order permutations for multiple choice (e.g. MMLU). Can improve results by de-biasing models's preferences for arbitrary answer orderings. Best results with
      'all' but very slow. 'cyclic' is a good compromise. single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations' (default: single)
```

The permutation options will automatically duplicate the inputs, but with the answer orders permuted (e.g. With "A" now being "cranial", "B" being "pleural" etc.)
This improves results for when we have very small amounts of supervised data (or zeroshot). 
The code will automatically marginalize across results for evaluation permutations for you, in the case you use --multiple_choice_eval_permutations option `cyclic` or `all`.
More details on the permutation de-biasing can be found in the appendix of [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/pdf/2208.03299.pdf).

If you pass `--write_results`, Atlas will write its predicted answers to file, with the following format:

```json
{
  "question": "the prompt-template applied input",
  "generation": "answer letter choice with highest probability after marginalizing across permutations",
  "choice_probs": "the probability of each answer choice (normalized over total answer options)",
  "all_probs": "the un-marginalized answer probabilities from all the answer order permutations",
  "permutations": ["the list of prediction objects for each permutation of the answer ordering"]
}
```

#### MMLU

A dedicated ReadMe is available for running MMLU experiments [here](./example_scripts/mmlu/README_MMLU.md). 
There is a tool to download and preprocess the MMLU data, and example scripts for running each of the experimental settings that we explore with MMLU are available `examples/mmlu`.
These are documented in detail in the MMLU Dedicated Readme.

</details>


<details>
<summary>
<h4 name="fever-task">FEVER Fact Verification</h4>
</summary>

Atlas can be trained to classify textual claims as "SUPPORTED", "REFUTED" or "NOT_ENOUGH_INFO" by a corpus, such as for the FEVER task  by using `--task fever` to `train.py` or `evaluate.py`.
	
You can download the FEVER data by calling the following script:

```bash
python preprocessing/prepare_fever.py --output_directory ${DATA_DIR} 
```

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format:

```python
{
  "claim": "the claim to assess", 
  "label": "either 'SUPPORTS', 'REFUTES' or 'NOT ENOUGH INFO'",
   ... # you can have other fields you want to keep around for ease of analysis, but they wont actually be used
}
```
Atlas will automatically process these instances, and format them for input as `question: {claim} answer: <extra_id_0>` and the output as `<extra_id_0> {true, false or maybe}`.
If you pass `--write_results`, Atlas will write its predicted labels to file.
Atlas will log the following evaluation metrics for open domain QA during its evaluation loop: 

* `accuracy`:  how many claims were correctly classified by the model.

</details>

<details>
<summary>
<h4 name="kilt-task">KILT</h4>
</summary>

Atlas can be trained to perform KILT tasks by using `--task kilt` to `train.py` or `evaluate.py`.

KILT data can be obtained from [here](https://github.com/facebookresearch/KILT)

Train/validation/test data for this task should consist of jsonl files, which should be passed to `train.py`  as `--train_data train_file_1.jsonl train_file_2.jsonl`, and `--eval_data eval_file_1.jsonl eval_file_2.jsonl` etc.
Files should have one JSON instance per line with the following format (i.e. the codebase will accept the KILT format directly):
```python
{'id': # original data point id if available otherwise unique id
 'input': # question / claim / sentence / etc
 'output': [ # each element might contain an answer, a provenance or both
    {
    'answer': # answer in textual form
    'provenance': [
        # evidence set for the answer from the KILT ks
        {
            'wikipedia_id':  # *mandatory* 
            'title': 
            'section': 
            'start_paragraph_id': 
            'start_character': 
            'end_paragraph_id':
            'end_character': 
            'bleu_score': # wrt original evidence
            'meta': # dataset/task specific
        }
        ] 
      }
    ]
 'meta': # dataset/task specific
 }
```
Atlas will automatically process these instances appropriately, into Atlas] query inputs based on the `input` field and target generations based on the `answer` fields

If you pass `--write_results`, Atlas will write its predicted labels to file.

Atlas will log the following evaluation metrics for open domain QA during its evaluation loop: 
* `accuracy`:  how often generations exactly match the reference
* `exact_match`:  how often generations exactly match the reference, with open-domain QA normalization applied
* `f1`:  the token level f1 score overlap between the generation and reference

</details>


## Retrieval and Index Details

The following section gives more details on retrieval and indices.

As briefly mentioned in the introduction, retrieval is handled in the Atlas code by taking advantage of the parallel nature of training modern large neural networks.
Specifically, for all modern training on GPUs, training (and inference) requires several GPU workers to be available for parallel computation.

Atlas makes use of this already-existing distributed setup.
It shards its retrieval index (passages + embedded passages) into N equally sized shards, with one shard per GPU worker. 
By default, retrieval is performed entirely using exact search on GPU using pytorch, (no approximate search using FAISS), which is still fast because the search is parallelized across all the GPU workers (assuming there are enough GPUs).

Knowing the mechanics of how retrieval is accomplished should not be needed to run the codebase, but might be useful to adapt the codebase to specific ends.
We thus include a brief description below. For simplicity, we'll assume we're performing open-domain question answering, and using a per_gpu_batch_size of 1, and assume we have W GPU workers, and N total passages in the retrieval index.

There are two high level functions that the retriever needs to do:
<details>
<summary>
1. Build/Refresh Embeddings
</summary>

Building or refreshing the index involves calculating embeddings for every passage in the retrieval index, which are then saved in memory, which allows for fast computation of maximum inner products/nearest neighbors when doing retrieval later. 
(Assume we have W GPU workers, and N total passages in the retrieval index)

Recall that each GPU worker has a shard of N/W passages.
Passage embedding calculations is quite simple, and proceeds as follows:
* We halt any model training going on
* Each worker calculates embeddings for its shard of passages in parallel, iterating over them in batches, saving them in large torch tensor (faiss support is also available, but discussed later). 
* When all the workers have finished embedding their shard, we can continue with model training.

N may be very large (10-100M), so embedding all the passages is quite slow. 
However, because we can parallelize it across our workers, it can be relatively fast if we have enough GPUS. None-the-less, for large indices, this can still be quite slow.
We have functionality to [save indices](#index-saving-and-loading) to disk to avoid excessive index building.
Moreover, as the retriever gets trained, the cached embeddings become out-of-date, or "stale", and need to be recalculated, which incurs even more cost.  See [Strategies for dealing with stale indices](#strategies-for-dealing-with-stale-indices) for ways to reduce/avoid the need for frequent index refreshes.
</details>

<details>
<summary>
2. Perform Distributed Retrieval
</summary>

Atlas performs retrieval-in-the-loop: i.e. a retrieval call is used as part of the forward pass.
Here, we'll briefly the describe the steps in a forward pass of Atlas, including how retrieval is accomplished:
(Assume we have W GPU workers, and N total passages in the retrieval index, and for simplicity, assume a training per gpu batch size of 1, and a question-answering task).

* Each worker has a question, which it embeds into a query vector.
* An all-gather is performed, which results in each worker having a copy of all the W query vectors (i.e. a query vector for all the questions in the total minibatch)
* Each worker then performs a maximum inner product search on GPU for its shard for all the query vectors in the batch.
* The top K results from each workers shard for each query are then sent back to the GPU that embedded that query, via a gather.
* This results in each worker with the top W * K results for its query, from which the true top K results are selected.
* The retrieval is now complete, and a standard distributed-data-parallel forward pass can continue (i.e. run the model forward pass, calculate gradients, aggregate gradients across workers, and update the parameters.)

</details>

### Flat vs Faiss

There are two index modes implemented for Atlas. 
By default, we perform retrieval using an exact search ('Flat') index, where retrieval is performed on GPU using pure pytorch.
We also support a [FAISS](https://github.com/facebookresearch/faiss) mode, which is useful for saving GPU memory for extremely large indices, or where GPU memory is very restricted.
FAISS is a library for fast approximate nearest neighbor search. Our retrieval is on GPU, so we do not usually require further search acceleration, but faiss can be used for compressing the size of an index in memory, which may be of use for very large indices.

The mode to use is specified by `--index_mode {"flat"|"faiss"}`. 
For most use cases, the `flat` index will be sufficient and likely preferable. 

If using the faiss index, users should specify what kind of faiss index to use, using the following options:

```
  --faiss_index_type {ivfflat,flat,ivfsq,ivfpq,pq}
      IVFFlat, IndexFlatIP, IVFScalarQuantizer, IndexPQ or IndexIVFPQ with faiss-gpu (default: flat)
  --faiss_code_size FAISS_CODE_SIZE
      Parameter for PQ/SQ quantization (default: None)
```

A good default if using a faiss index is to use `--faiss_index_type ivfpq --faiss_code_size 16`. This will use an IVF-PQ index with the number of IVF clusters set to the square root of the number of embeddings per shard, and PQ code size of 16. More details on this index structure can be found in the faiss documentation [FAISS](https://github.com/facebookresearch/faiss).

### Index Saving and Loading

Indices (passage and embeddings shards) can be saved to disk and loaded in, to avoid recomputing them.
See [above](#prebuilt-indices) for some downloadable indices.

Index saving can be switched on using `--save_index_path {path/to/directory/save/index/in}`, which will create a directory,  
and save each worker's embedding shard to index (as a pytorch tensor on disk) and passages shard (as a pickle file).

To load an index, pass `--load_index_path {path}`, which will load the index at the specified path.

Saving and loading works with both `flat` and `faiss` modes.

In order to easily load an index when using a different number of workers from the index that created it, we can configure `--save_index_n_shards N`, which will save the index into N shards (for example if we have 32 workers, we can pass `--save_index_n_shards 128` to save the index as 128 shards to disk). 
When we try to load the index again, for example with 64 workers, the code will figure out it should load 2 saved files per worker. (Note: this functionality only works with `flat` indices - for faiss indices, you can only load indices where the number of workers is the same as when it was saved to disk).

### Strategies for dealing with stale indices

As the retriever is trained, the passage embeddings stored in memory become stale. 
This affects the accuracy of retrieval, and, over long periods of time, may lead to suboptimal training or instability.
Atlas has three methods that can combat this

1. <b name="#index-refresh">Index Refresh</b>: The simplest and most expensive option is to recompute the embeddings using the up-to-date retriever embedder. The index refresh rate schedule is controlled by the `--refresh_index` argument. format: `startstep-endstep:refreshrate,` e.g. `--refresh_index 0-1000:500,1000-10000:1000` will refresh the index every 500 steps for the first 1000 steps, and then every 1000 steps from step 1000 to 10000. You can also just pass in a single number e.g. `--refresh_index 100` will refresh the index every 100 steps. Pass `--refresh_index -1` to never refresh. We use this setting for large datasets and pretraining. 
2. <b name="#overretrieve-with-reranking">Over-Retrieve with Reranking</b>: Here, instead of refreshing the index, we can retrieve the top L passages (where L > K), and then, rerank these L passages using the up-to-date embedder on-the-fly, and pass the top K of these. This works well if the true top K are indeed contained in the stale top L. To use this pass `--retrieve_with_rerank` and specify `--n_to_rerank_with_retrieve_with_rerank L`. This method can be used in conjunction with index refreshing, to reduce staleness between refreshes.
3.  <b name="#query-Side-finetuning">Query-Side Finetuning</b>: To avoid stale-ness, we can keep the passage embedder of the retriever fixed, and only train the query embedder. This method will sacrifice retriever performance if there is lots of training data, but works well in few-shot settings. To enable this mode, pass `--query_side_retriever_training`. Note: usually we use parameter sharing for the passage and query encoder of the retriever - this mode is the exception, where we break the parameter tying to keep the passage encoder fixed.

### Retrieve-only mode

Atlas can be used purely in a retrieval mode at evaluation time. 
This can be useful for users who want a fast, scalable, easy to launch GPU-enabled dense retriever.

In this mode, (which only works with `evaluate.py`) no reader language model gets loaded, and the script will perform retrieval, and then write retrieval results to file if the `--write_results` flag has been passed.

To use this mode, pass `--retrieve_only` to `evaluate.py`.
There is an example of NaturalQuestions retrieval using this mode in `examples/nq/retrieve_only.sh`.

### Using pre-retrieved or cached passages

In some cases, users may have already performed retrieval and want to cache the retrieved results for their dataset, or know a priori the most relevant passages, and thus do not need to perform retrieval.

In these cases, Atlas can be forced to use user-specified passages per input instance, rather than retrieve, by 1) passing the `--use_file_passages` flag and 2) including a json field `passages` in the train/eval files they pass in, with the following format (e.g for the `qa` task)

<details>
<summary>
(click to expand to see example)
</summary>

```python
{
  "question": "where is the bermuda triangle",
  "answers": ["Western part of the North Atlantic Ocean"],
  "passages": [
    {
      "text": "text of first passage",
      "title": "title of  first passage",
      "id": "id of first passage"
      ... # other fields can be here but wont be used
    },
    {
      "text": "text of second passage",
      "title": "title of  second passage",
      "id": "id of second passage"
    },
    ... # more passages if you like
  ]
}
```

</details>

## Other features

The following are other features that Atlas provides for advanced users:

### Closed book mode

Atlas can be run as a standard non-retrieval-augmented T5 model, often referred to as "closed-book" in the literature. This is useful for running baseline experiments, and checking that your model does indeed benefit from retrieval-augmentation for your task. Pass the `--closed_book` argument to do closed-book training and ignore the retrieved passages.

### Specifying formats

Format strings can be injected for greater formatting control of how the inputs get presented to the Atlas model:

```
  --encoder_format ENCODER_FORMAT
    format string for reader's encoder preprocessing (default: "{query} title: {title} context: {text}")
  --retriever_format RETRIEVER_FORMAT
    format string for retriever's encoder preprocessing (default: "{title} {text}")
```

For example, passing `--encoder_format "{query} text: {text}"` wouldn't pass the retrieved passages' titles to the reader model.


### Implementing your own task

To implement a new task for Atlas, there are two options: the easiest is to preprocess or format your task to be compatible using one of the already implemented tasks (the `base` task should support almost all potential use cases).

The other is to implement your own task under `src/tasks/your_task_name.py` and import it under `src/tasks/__init__.py`.

See the `src/tasks/qa.py` for an example. 

The `process` function takes the raw parsed, jsonl-objects passed to --train_data or --eval_data, and should return a dict with `{query: "query to pass to Atlas", "target": "target string", "passages": [list of gold retrieved passages, can be empty]}`

The `evaluate` function takes a predicted generation and references for a task, and return a dict of task-specific evaluation scores, which the codebase will average across evaluation instances.

## Full list of command line flags:

<details>
<summary>
Click to Expand
</summary>

```
usage: train.py/evaluate.py [-h] [--name NAME] [--checkpoint_dir CHECKPOINT_DIR] [--model_path MODEL_PATH] [--per_gpu_batch_size PER_GPU_BATCH_SIZE] [--per_gpu_embedder_batch_size PER_GPU_EMBEDDER_BATCH_SIZE] [--local_rank LOCAL_RANK]
                [--main_port MAIN_PORT] [--seed SEED] [--log_freq LOG_FREQ] [--eval_freq EVAL_FREQ] [--save_freq SAVE_FREQ] [--train_data TRAIN_DATA [TRAIN_DATA ...]] [--eval_data EVAL_DATA [EVAL_DATA ...]] [--write_results]
                [--dont_write_passages] [--load_index_path LOAD_INDEX_PATH] [--save_index_path SAVE_INDEX_PATH] [--save_index_n_shards SAVE_INDEX_N_SHARDS] [--index_mode {flat,faiss}] [--faiss_index_type {ivfflat,flat,ivfsq,sq,pq}]
                [--faiss_code_size FAISS_CODE_SIZE] --reader_model_type
                {t5-small,t5-base,t5-large,t5-3b,t5-11b,google/t5-v1_1-base,google/t5-v1_1-large,google/t5-v1_1-xl,google/t5-v1_1-xxl,google/t5-base-lm-adapt,google/t5-large-lm-adapt,google/t5-xl-lm-adapt,google/t5-xxl-lm-adapt}
                [--text_maxlength TEXT_MAXLENGTH] [--target_maxlength TARGET_MAXLENGTH] [--n_context N_CONTEXT] [--passages PASSAGES [PASSAGES ...]] [--max_passages MAX_PASSAGES] [--retriever_model_path RETRIEVER_MODEL_PATH]
                [--retrieve_only] [--train_retriever] [--use_file_passages] [--retriever_n_context RETRIEVER_N_CONTEXT] [--gold_score_mode {evalnormsum,loop,ppmean,emdr,pdist,adist}] [--closed_book]
                [--temperature_score TEMPERATURE_SCORE] [--temperature_gold TEMPERATURE_GOLD] [--compute_crossattention_stats] [--filtering_overretrieve_ratio FILTERING_OVERRETRIEVE_RATIO]
                [--freeze_retriever_steps FREEZE_RETRIEVER_STEPS] [--query_side_retriever_training] [--retrieve_with_rerank] [--n_to_rerank_with_retrieve_with_rerank N_TO_RERANK_WITH_RETRIEVE_WITH_RERANK]
                [--decoder_format DECODER_FORMAT] [--decoder_prompt_format DECODER_PROMPT_FORMAT] [--encoder_format ENCODER_FORMAT] [--retriever_format RETRIEVER_FORMAT] [--generation_max_length GENERATION_MAX_LENGTH]
                [--generation_min_length GENERATION_MIN_LENGTH] [--generation_length_penalty GENERATION_LENGTH_PENALTY] [--generation_num_beams GENERATION_NUM_BEAMS] [--task {base,mlm,lm,multiple_choice,kilt,section,fever,qa}]
                [--mlm_noise_density MLM_NOISE_DENSITY] [--mlm_mean_noise_span_length MLM_MEAN_NOISE_SPAN_LENGTH] [--min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE] [--min_lm_context_ratio MIN_LM_CONTEXT_RATIO]
                [--max_lm_context_ratio MAX_LM_CONTEXT_RATIO] [--qa_prompt_format QA_PROMPT_FORMAT] [--multiple_choice_num_options MULTIPLE_CHOICE_NUM_OPTIONS] [--multiple_choice_train_permutations {single,cyclic,all}]
                [--multiple_choice_eval_permutations {single,cyclic,all}] [--warmup_steps WARMUP_STEPS] [--total_steps TOTAL_STEPS] [--scheduler_steps SCHEDULER_STEPS] [--accumulation_steps ACCUMULATION_STEPS] [--dropout DROPOUT]
                [--lr LR] [--lr_retriever LR_RETRIEVER] [--clip CLIP] [--scheduler {linear,cosine,fixed}] [--weight_decay WEIGHT_DECAY] [--save_optimizer] [--epsilon EPSILON] [--alpha ALPHA] [--beta2 BETA2]
                [--refresh_index REFRESH_INDEX] [--shuffle] [--precision {fp16,fp32,bf16}] [--shard_optim] [--shard_grads] [--use_gradient_checkpoint_reader] [--use_gradient_checkpoint_retriever]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the experiment - also used as directory name (default: experiment_name)
  --checkpoint_dir CHECKPOINT_DIR
                        models are saved here (default: ./checkpoint/)
  --model_path MODEL_PATH
                        Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever) (default: none)
  --per_gpu_batch_size PER_GPU_BATCH_SIZE
                        Batch size per GPU/CPU for training. (default: 1)
  --per_gpu_embedder_batch_size PER_GPU_EMBEDDER_BATCH_SIZE
                        Embedder's batch size per GPU. (default: 512)
  --local_rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
  --main_port MAIN_PORT
                        Main port (for multi-node jobs) (default: -1)
  --seed SEED           random seed for initialization (default: 0)
  --log_freq LOG_FREQ   log train stats <log_freq> steps during training (default: 100)
  --eval_freq EVAL_FREQ
                        evaluate model every <eval_freq> steps during training (default: 500)
  --save_freq SAVE_FREQ
                        save model every <save_freq> steps during training (default: 5000)
  --train_data TRAIN_DATA [TRAIN_DATA ...]
                        list of space-separated paths to jsonl-formatted train sets (default: [])
  --eval_data EVAL_DATA [EVAL_DATA ...]
                        list of space-separated paths to jsonl-formatted evaluation sets (default: [])
  --write_results       save evaluation results to file (default: False)
  --dont_write_passages
                        if writing results, passages can take up a lot of space, pass this flag not to write passages as part of dumped results (default: False)
  --load_index_path LOAD_INDEX_PATH
                        path for loading the index, passage embeddings and passages (default: None)
  --save_index_path SAVE_INDEX_PATH
                        path for saving the index and/or embeddings (default: None)
  --save_index_n_shards SAVE_INDEX_N_SHARDS
                        how many shards to save an index to file with. Must be an integer multiple of the number of workers. (default: 128)
  --index_mode {flat,faiss}
                        Use flat torch index or a faiss index for retrieving the k nearest neighbors (default: flat)
  --faiss_index_type {ivfflat,flat,ivfsq,sq,pq}
                        IVFFlat, IndexFlatIP, IVFScalarQuantizer, ScalarQuantizer or IndexPQ with faiss-gpu (default: flat)
  --faiss_code_size FAISS_CODE_SIZE
                        Parameter for PQ quantization (default: None)
  --reader_model_type {t5-small,t5-base,t5-large,t5-3b,t5-11b,google/t5-v1_1-base,google/t5-v1_1-large,google/t5-v1_1-xl,google/t5-v1_1-xxl,google/t5-base-lm-adapt,google/t5-large-lm-adapt,google/t5-xl-lm-adapt,google/t5-xxl-lm-adapt}
                        t5 Architecture for reader FID model, e.g. google/t5-xl-lm-adapt (default: None)
  --text_maxlength TEXT_MAXLENGTH
                        maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated. (default: 200)
  --target_maxlength TARGET_MAXLENGTH
                        Maximum length of target outputs in tokens when training the model. Targets longer than this will be truncated. No truncation if -1 (default: None)
  --n_context N_CONTEXT
                        number of top k passages to pass to reader (default: 1)
  --passages PASSAGES [PASSAGES ...]
                        list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path (default: None)
  --max_passages MAX_PASSAGES
                        maximum number of passages to index. -1 to read all passages in passage files (default: -1)
  --retriever_model_path RETRIEVER_MODEL_PATH
                        path to contriever model to init from (overridden if passing a value to --model_path (default: facebook/contriever)
  --retrieve_only       Pass this to prevent loading a reader, and only run retrieval evaluation (default: False)
  --train_retriever     Pass to train retriever as well as reader (default: False)
  --use_file_passages   uses passages in "passages" field in train or eval jsonl files rather than retrieving passages (default: False)
  --retriever_n_context RETRIEVER_N_CONTEXT
                        number of top k passages to use to train the retriever with (default: 5)
  --gold_score_mode {evalnormsum,loop,ppmean,emdr,pdist,adist}
                        retriever training method. `pdist` is the name used in the paper for `ppmean`. `adist` is the name used in the paper for `evalnormsum` (default: ppmean)
  --closed_book         Don't use retrieval - reduces to T5. Overrides n_context, n_context_retriever and encoder_format if they are set (default: False)
  --temperature_score TEMPERATURE_SCORE
                        softmax temperature for retriever (default: 0.01)
  --temperature_gold TEMPERATURE_GOLD
                        softmax temperature for target distribution for retriever distillation (default: 0.01)
  --compute_crossattention_stats
  --filtering_overretrieve_ratio FILTERING_OVERRETRIEVE_RATIO
                        if filtering, over-retrieve the topK by this factor, and then filter out undesirable results. Useful, Set to 1 only if using a task that doesn't filter retrieved results (default: 2)
  --freeze_retriever_steps FREEZE_RETRIEVER_STEPS
                        freezes retriever for n steps (default: -1)
  --query_side_retriever_training
                        pass to enable query-side finetuning of retriever (unties the parameters of the contriever encoder's passage and query encoders, and freezes the passage encoder. Useful to avoid index refreshes. (default: False)
  --retrieve_with_rerank
                        pass this to enable reranking with fresh passage encoder for retriever (default: False)
  --n_to_rerank_with_retrieve_with_rerank N_TO_RERANK_WITH_RETRIEVE_WITH_RERANK
                        n passages to rerank when passing --retrieve_with_rerank. Higher is slower but more accurate. Recommend 64-128 (default: 128)
  --decoder_format DECODER_FORMAT
                        format for decoder, model will be train on the format and evaluation will be performed with the format contrary to the decoder_prompt_format option (default: None)
  --decoder_prompt_format DECODER_PROMPT_FORMAT
                        format for decoder prompting, for instance "what is the answer to {query}:" (default: None)
  --encoder_format ENCODER_FORMAT
                        format string for reader's encoder preprocessing (default: {query} title: {title} context: {text})
  --retriever_format RETRIEVER_FORMAT
                        format string for retriever's encoder preprocessing (default: {title} {text})
  --generation_max_length GENERATION_MAX_LENGTH
  --generation_min_length GENERATION_MIN_LENGTH
  --generation_length_penalty GENERATION_LENGTH_PENALTY
  --generation_num_beams GENERATION_NUM_BEAMS
  --task {base,mlm,lm,multiple_choice,kilt,section,fever,qa}
                        Task performed by the model. Used to setup preprocessing, retrieval filtering, evaluations, etc. (default: None)
  --mlm_noise_density MLM_NOISE_DENSITY
                        how much of an input text should be masked by masking spans (default: 0.15)
  --mlm_mean_noise_span_length MLM_MEAN_NOISE_SPAN_LENGTH
                        average length of an MLM masking span (default: 3)
  --min_words_per_lm_instance MIN_WORDS_PER_LM_INSTANCE
                        Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section Generation (default: None)
  --min_lm_context_ratio MIN_LM_CONTEXT_RATIO
                        Splits text into two segments for language modelling.' 'Left segment is conditioning context, right segment is for generating.' 'The left segment must be more than min_lm_context_ratio of the right segment
                        (default: 0.5)
  --max_lm_context_ratio MAX_LM_CONTEXT_RATIO
                        Splits text into two segments for language modelling.' 'Left segment is conditioning context, right segment is for generating.' 'The left segment must be less than max_lm_context_ratio of the right
                        segment (default: 0.5)
  --qa_prompt_format QA_PROMPT_FORMAT
                        How to format question as input prompts when using --task qa (default: question: {question} answer: <extra_id_0>)
  --multiple_choice_num_options MULTIPLE_CHOICE_NUM_OPTIONS
                        How many choice options for multiple choice QA (MMLU is 4) (default: 4)
  --multiple_choice_train_permutations {single,cyclic,all}
                        Whether to train with answer order permutations When training on multiple choice (e.g. MMLU). Can improve results by de-biasing models's preferences for arbitrary answer orderings. Recommend training with 'all'.
                        single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations' (default: single)
  --multiple_choice_eval_permutations {single,cyclic,all}
                        Whether to evaluate with answer order permutations for multiple choice (e.g. MMLU). Can improve results by de-biasing models's preferences for arbitrary answer orderings. Best results with 'all' but very slow.
                        'cyclic' is a good compromise. single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations' (default: single)
  --warmup_steps WARMUP_STEPS
                        number of learning rate warmup steps (default: 1000)
  --total_steps TOTAL_STEPS
                        total number of training steps (default: 1000)
  --scheduler_steps SCHEDULER_STEPS
                        total number of step for the scheduler, if None then scheduler_total_step = total_step (default: None)
  --accumulation_steps ACCUMULATION_STEPS
                        gradient accumulation (default: 1)
  --dropout DROPOUT     dropout rate (default: 0.1)
  --lr LR               learning rate (default: 0.0001)
  --lr_retriever LR_RETRIEVER
                        learning rate for retriever (default: 1e-05)
  --clip CLIP           gradient clipping (default: 1.0)
  --scheduler {linear,cosine,fixed}
                        learning rate schedule to use (default: cosine)
  --weight_decay WEIGHT_DECAY
                        amount of weight decay to apply in training (default: 0.1)
  --save_optimizer      Pass flag to save optimizer state in saved checkpoints (default: False)
  --epsilon EPSILON     adamw epsilon value (default: 1e-06)
  --alpha ALPHA         adamw alpha value (default: 1.0)
  --beta2 BETA2         adamw beta2 value (default: 0.999)
  --refresh_index REFRESH_INDEX
                        index refresh schedule. format: startstep-endstep:refreshrate,startstep-endstep:refreshrate e.g. --refresh_index 0-100:10,100-1000000:500 will refresh the index every 10 steps for the first 100 steps, and then
                        every 500 steps from step 100 to 1M.Syntactic Sugar for a fixed schedule: can just pass in a single number e.g. --refresh_index 100 will refresh the index every 100 steps. -1 to never refresh. (default:
                        0-1000000:1000000)
  --shuffle             shuffle data for training (default: False)
  --precision {fp16,fp32,bf16}
                        numerical precision - recommend bf16 if available, fp16 likely to be unstable for training (default: fp32)
  --shard_optim         train-time memory optimization: shards optimizer state over available GPUs using sharded data parallel, recommended for larger models (default: False)
  --shard_grads         train-time memory optimization: shards gradients over available GPUs using sharded data parallel, recommended for larger models (default: False)
  --use_gradient_checkpoint_reader
                        use gradient checkpointing in the reader (default: False)
  --use_gradient_checkpoint_retriever
                        use gradient checkpointing for retriever (default: False)
```

</details>

## Citing

To cite this work, please use the following bibtex:
```
@article{izacard_few-shot_2022,
	title = {Few-shot {Learning} with {Retrieval} {Augmented} {Language} {Models}},
	url = {http://arxiv.org/abs/2208.03299},
	publisher = {arXiv},
	author = {Izacard, Gautier and Lewis, Patrick and Lomeli, Maria and Hosseini, Lucas and Petroni, Fabio and Schick, Timo and Dwivedi-Yu, Jane and Joulin, Armand and Riedel, Sebastian and Grave, Edouard},
	year = {2022},
}
```

## License

### Code License:

The majority of the Atlas code is licensed under [CC-BY-NC](./LICENSE), however portions of the project are available under separate license terms: huggingface transformers is licensed under the [Apache 2.0 license](https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE), which covers `src/modeling_bert.py` and `src/modeling_t5.py`.

### Data License:

The wikipedia-derived data used in the repository, such as the corpora and indices available from `download_corpus.py` and `download_index.py` are licensed according to [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). 
