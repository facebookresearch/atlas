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

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/templama_data/temp_lama.train.${YEAR}.jsonl"
EVAL_FILES="${DATA_DIR}/data/templama_data/temp_lama.valid.${YEAR}.jsonl ${DATA_DIR}/data/templama_data/temp_lama.test.${YEAR}.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PASSAGES="${DATA_DIR}/copora/wiki/enwiki-dec${YEAR}/text-list-100-sec.jsonl ${DATA_DIR}/copora/wiki/enwiki-dec${YEAR}/infobox.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-templama-${YEAR}
PRECISION="fp32" # "bf16"

srun python train.py \
    --shuffle \
    --train_retriever --query_side_retriever_training\
    --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever\
    --precision ${PRECISION} \
    --shard_optim --shard_grads \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index -1 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 4e-5 --lr_retriever 4e-5 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 20 --retriever_n_context 20 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 20 \
    --log_freq 4 \
    --total_steps 100 \
    --warmup_steps 20 \
    --save_freq 100 \
    --main_port $port \
    --write_results \
    --task qa \
    --index_mode flat \
    --passages ${PASSAGES}\
    --qa_prompt_format "{question}"
