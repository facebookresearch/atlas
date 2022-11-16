#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --job-name=templama
#SBATCH --output=run_dir/%A.out
#SBATCH --error=run_dir/%A.err
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb

size=xl
DATA_DIR='/checkpoint/plewis/atlas_opensourcing/'
YEAR=${1:-"2017"}
MODEL_TO_EVAL=''# Model finetuned by examples/templama/train.sh

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/templama_data/temp_lama.train.${YEAR}.jsonl"
EVAL_FILES="${DATA_DIR}/data/templama_data/temp_lama.valid.${YEAR}.jsonl ${DATA_DIR}/data/templama_data/temp_lama.test.${YEAR}.jsonl"
PASSAGES="${DATA_DIR}/copora/wiki/enwiki-dec${YEAR}/text-list-100-sec.jsonl ${DATA_DIR}/copora/wiki/enwiki-dec${YEAR}/infobox.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-templama-${YEAR}
PRECISION="fp32" # "bf16"

srun python evaluate.py \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 32 --target_maxlength 32 \
    --gold_score_mode "ppmean" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 384 \
    --model_path ${MODEL_TO_EVAL} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 20 --retriever_n_context 20 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --passages ${PASSAGES}\
    --write_results
    --qa_prompt_format "{question}"
