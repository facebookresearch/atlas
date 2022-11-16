#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --job-name=mlm
#SBATCH --output=run_dir/%A.out
#SBATCH --error=run_dir/%A.err
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --mem=480GB

# This example script will perform ATLAS-style MLM pretraining on wikipedia 2018
# First it will download the data, then shuffle and split into train/dev/test splits, then call atlas MLM training
# Note how the training data is the data to denoise AND the corpus to retrieve from.

size=xl
DATA_DIR="/checkpoint/plewis/atlas_opensourcing_check/"

# download the Wikipedia 2018 corpus:
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR}

# Prepare train/dev/test data from corpus:
TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"

shuf ${TEXTS} > "${TEXTS}.shuf"
head -n 2000 "${TEXTS}.shuf" | head -n 1000 > "${TEXTS}.shuf.test"
head -n 2000 "${TEXTS}.shuf" | tail -n 1000 > "${TEXTS}.shuf.valid"
tail -n +2000 "${TEXTS}.shuf" > "${TEXTS}.shuf.train"

shuf ${INFOBOXES} > "${INFOBOXES}.shuf"
head -n 2000 "${INFOBOXES}.shuf" | head -n 1000 > "${INFOBOXES}.shuf.test"
head -n 2000 "${INFOBOXES}.shuf" | tail -n 1000 > "${INFOBOXES}.shuf.valid"
tail -n +2000 "${INFOBOXES}.shuf" > "${INFOBOXES}.shuf.train"


port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILES="${TEXTS}.shuf.train ${INFOBOXES}.shuf.train"
EVAL_FILES="${TEXTS}.shuf.valid ${INFOBOXES}.shuf.valid ${TEXTS}.shuf.test ${INFOBOXES}.shuf.test"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-wiki-mlm-pretrain
PRECISION="fp32" # "bf16"


srun python train.py \
    --retrieve_with_rerank --n_to_rerank_with_retrieve_with_rerank 100 \
    --train_retriever --gold_score_mode "pdist" \
    --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever \
    --shard_grads --shard_optim \
    --precision ${PRECISION} \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index 1000 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --passages ${TRAIN_FILES} \
    --target_maxlength 64 \
    --dropout 0.1 \
    --lr 1e-4 --lr_retriever 1e-5\
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 384 \
    --model_path none \
    --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 20 --retriever_n_context 20 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --save_freq 5000 --eval_freq 1000 --log_freq 100 \
    --total_steps 10000 \
    --warmup_steps 1000 \
    --main_port $port \
    --min_words_per_lm_instance 10 \
    --task "mlm" \
    --mlm_noise_density 0.15 \
    --mlm_mean_noise_span_length 3 
