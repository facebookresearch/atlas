# MMLU experiments:

This readme describes how to run ATLAS MMLU experiments like those in the [Fewshot learning With Retrieval-Augmented Language Models](https://arxiv.org/pdf/2208.03299.pdf) paper.
There are example scripts for running training for zeroshot, 5-shot, 5-shot-multi and full experiments in `<project_root>/examples/mmlu`

## Getting data:

Download and preprocess the data using the preprocessing data:

```bash
DATA_DIR=path/to/data
python preprocessing/prepare_mmlu.py --output_directory ${DATA_DIR}
```

This script downloads, parses and creates train, validation and test files for MMLU, and saves them to ${MY_DATA_DIR}/data/mmlu_data

We consider 3 tasks (zeroshot will use the 5-shot):
* 5-shot: learn a model with 5 examples for each domain. 
* 5-shot-multitask: Learn a single model using the combination of 5 examples from each domain.
* full: Learn a single model using training data from MMLU's auxialluary datasets, plus the training data from 5-shot-multitask.
* (for zeroshot, we can use the 5-shot multitask settting with zero training steps)

In each case, overall test accuracy would be the micro average over each domains' test set (as defined by the orginal authors).

The script will download the data, and create the following directory structure:
```bash
├── data.tar # original data
├── 5-shot 
│   ├── combined_test.jsonl
│   ├── combined_valid.jsonl
│   ├── individual_test
│   │   ├── {domain}.test.jsonl
│   ├── individual_train
│   │   ├── {domain}.5-shot-train.jsonl
│   └── individual_valid
│       ├── {domain}.val.jsonl
├── 5-shot-multitask 
│   ├── combined_test.jsonl
│   ├── combined_valid.jsonl
│   ├── individual_test
│   │   ├── {domain}.test.jsonl
│   ├── individual_valid
│   │   ├── {domain}.val.jsonl
│   └── train.jsonl
└── full
    ├── auxillary_valid.jsonl
    ├── combined_test.jsonl
    ├── combined_valid.jsonl
    ├── individual_test
    │   ├── {domain}.test.jsonl
    ├── individual_valid
    │   ├── {domain}.val.jsonl
    └── train.jsonl
```
* For 5-shot, train models 5-shot/individual_train/{domain}.5-shot-train.jsonl and test on 5-shot/individual_test/{domain}.test.jsonl
* For 5-shot-multitask, train models 5-shot-multitask/train.jsonl and test on 5-shot-multitask/combined_test.jsonl
* For the full data task, train models full/train.jsonl and test on full/combined_test.jsonl

## Running experiments:

Run the following scripts to launch training and do evaluation:

### Zeroshot experiments

Run the following command to run a zeroshot (evaluation without any finetuning) experiment:

<details>
<summary>
Expand to see script:
</summary>

```bash
# assumes 8 nodes x 8 A100 GPUs are available
size=xxl

# downloads pretrained ATLAS-xxl
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}  
# downloads a pre-built ATLAS-xxl index for wikipedia 2021
python preprocessing/download_index.py --model indices/atlas/wiki/${SIZE} --output_directory ${DATA_DIR}  

port=$(shuf -i 15000-16000 -n 1)
EVAL_FILES="${DATA_DIR}/data/mmlu_data/5-shot-multitask/combined_valid.jsonl ${DATA_DIR}/data/mmlu_data/5-shot-multitask/combined_test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-mmlu-zeroshot

srun python evaluate.py \
    --precision bf16 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 30 --retriever_n_context 30 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all\
    --multiple_choice_eval_permutations cyclic\
    --index_mode flat \
    --load_index_path ${PRETRAINED_INDEX}

```

</details>

Alternatively, if using slurm, run `sbatch examples/mmlu/mmlu-zeroshot.sh`.
After the script has run, results will be saved to `${SAVE_DIR}/${EXPERIMENT_NAME}/combined_test-step-1.jsonl`.

You can get the test scores as follows (using the wikipedia index - using a wikipedia + CC index instead would achieve the scores reported in the paper):

```bash 
$ python evaluation_scripts/evaluate_mmlu_predictions.py \
--predictions_path ${savedir}/${name}/combined_test-step-1.jsonl \
--gold_path ${MMLU_DATA_DIR}/5-shot-multitask/combined_test.jsonl

       category          Acc(%)   Debias Acc(%)
-----------------------------------------------
     humanities           37.60           44.14
       Soc Sci.           38.99           53.85
           STEM           30.47           37.62
          other           39.76           53.76
            all           36.87           47.09
```


### 5-shot experiments

Here, we must launch a training run for each of the 57 domains, then combine predictions in order to get final accuracy.
For each doamin, run the following:

<details>
<summary>
Expand to see script:
</summary>

```bash
# assumes 8 nodes x 8 A100 GPUs are available
size=xxl

# downloads pretrained ATLAS-xxl, if needed
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}  
# downloads a pre-built ATLAS-xxl index for wikipedia 2021, if needed
python preprocessing/download_index.py --model indices/atlas/wiki/${SIZE} --output_directory ${DATA_DIR}  

DOMAIN="abstract_algebra" # or any other of the 56 domains
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/mmlu_data/5-shot/individual_train/${DOMAIN}.5-shot-train.jsonl"
EVAL_FILES="${DATA_DIR}/data/mmlu_data/5-shot/individual_valid/${DOMAIN}.val.jsonl ${DATA_DIR}/data/mmlu_data/5-shot/individual_valid/${DOMAIN}.test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
SAVE_DIR=${DATA_DIR}/experiments/${SLURM_ARRAY_JOB_ID}-${size}-mmlu-5-shot/
EXPERIMENT_NAME=$DOMAIN

srun python train.py \
    --shuffle \
    --train_retriever --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever\
    --precision ${PRECISION} \
    --shard_optim --shard_grads \
    --temperature_gold 0.1 --temperature_score 0.1 \
    --refresh_index -1 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 5e-5 --lr_retriever 1e-5 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 30 --retriever_n_context 30 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 4 \
    --log_freq 4 \
    --total_steps 16 \
    --warmup_steps 4 \
    --save_freq 10000000000 \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all\
    --multiple_choice_eval_permutations cyclic\
    --index_mode flat \
    --query_side_retriever_training \
    --load_index_path ${PRETRAINED_INDEX}
```

</details>

Alternatively, if using slurm, run `sbatch azure/end2end/mmlu-5-shot.sh` to launch a job array that will run all 56 domains.

After a training script has run for each of the 56 domains, results for each domain will be saved to `${SAVE_DIR}/${DOMAIN}/${DOMAIN}-test-step-16.jsonl`.

You can get the test scores as follows:

```bash
$ python evaluation_scripts/evaluate_mmlu_predictions.py \
--predictions_path ${savedir} \
--gold_path ${DATA_DIR}/data/mmlu_data/5-shot-multitask/combined_test.jsonl \
--step 16 --split test
```


### 5-shot multitask experiments

Run the following command to run a 5-shot multitask experiment, which consists of finetuning on all of 5-shot training datasets at once:

<details>
<summary>
Expand to see script:
</summary>

```bash
# assumes 8 nodes x 8 A100 GPUs are available
size=xxl

# downloads pretrained ATLAS-xxl, if needed
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}  
# downloads a pre-built ATLAS-xxl index for wikipedia 2021, if needed
python preprocessing/download_index.py --model indices/atlas/wiki/${SIZE} --output_directory ${DATA_DIR}  

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/mmlu_data/5-shot-multitask/train.jsonl"
EVAL_FILES="${DATA_DIR}/data/mmlu_data/5-shot-multitask/combined_valid.jsonl ${DATA_DIR}/data/mmlu_data/5-shot-multitask/combined_test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-mmlu-5-shot-full

srun python train.py \
    --shuffle \
    --train_retriever --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever\
    --precision bf16 \
    --shard_optim --shard_grads \
    --temperature_gold 0.1 --temperature_score 0.1 \
    --refresh_index -1 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 5e-5 --lr_retriever 1e-5 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 30 --retriever_n_context 30 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 150 \
    --log_freq 4 \
    --total_steps 2000 \
    --warmup_steps 50 \
    --save_freq 10000000000 \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all\
    --multiple_choice_eval_permutations cyclic\
    --index_mode flat \
    --query_side_retriever_training \
    --load_index_path ${PRETRAINED_INDEX}

```

</details>

Alternatively, if using slurm, run `sbatch examples/mmlu/mmlu-5-shot-multi.sh`.
After the script has run, results will be saved to `${SAVE_DIR}/${EXPERIMENT_NAME}/combined_test-step-{step number}.jsonl`.

You can get the test scores as follows (deterimine step number from validation data):

```bash
$ python evaluation_scripts/evaluate_mmlu_predictions.py \
--predictions_path ${SAVE_DIR}/${EXPERIMENT_NAME}/combined_test-step-96.jsonl \
--gold_path ${DATA_DIR}/data/mmlu_data/5-shot-multitask/combined_test.jsonl

```

### full experiments

Run the following command to run a full data experiment:

<details>
<summary>
Expand to see script:
</summary>

```bash
# assumes 8 nodes x 8 A100 GPUs are available
size=xxl

# downloads pretrained ATLAS-xxl, if needed
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}  
# downloads a pre-built ATLAS-xxl index for wikipedia 2021, if needed
python preprocessing/download_index.py --model indices/atlas/wiki/${SIZE} --output_directory ${DATA_DIR}  

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/mmlu_data/full/train.jsonl"
EVAL_FILES="${DATA_DIR}/data/mmlu_data/full/combined_valid.jsonl ${DATA_DIR}/data/mmlu_data/full/combined_test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-mmlu-full
PRECISION="bf16" 

srun python train.py \
    --shuffle \
    --train_retriever --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever\
    --precision ${PRECISION} \
    --shard_optim --shard_grads \
    --temperature_gold 0.1 --temperature_score 0.1 \
    --refresh_index -1 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 5e-5 --lr_retriever 1e-5 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 30 --retriever_n_context 30 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 150 \
    --log_freq 4 \
    --total_steps 2000 \
    --warmup_steps 50 \
    --save_freq 10000000000 \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all\
    --multiple_choice_eval_permutations cyclic\
    --index_mode flat \
    --query_side_retriever_training \
    --load_index_path ${PRETRAINED_INDEX}
```

</details>


Alternatively, if using slurm, run `sbatch example_scripts/mmlu/mmlu-full.sh`.
After the script has run, results will be saved to `${SAVE_DIR}/${EXPERIMENT_NAME}/combined_test-step-{step number}.jsonl`.

You can get the test scores as follows (deterimine step number from validation data):

```bash 
$ python evaluation_scripts/evaluate_mmlu_predictions.py \
--predictions_path ${SAVE_DIR}/${EXPERIMENT_NAME}/combined_test-step-1800.jsonl \
--gold_path ${DATA_DIR}/data/mmlu_data/full/combined_test.jsonl

```