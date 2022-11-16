#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --job-name=nq
#SBATCH --output=run_dir/%A.out
#SBATCH --error=run_dir/%A.err
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --mem=480GB

size=xl
DATA_DIR='/checkpoint/plewis/atlas_opensourcing/'

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/nq_data/train.jsonl"
EVAL_FILES="${DATA_DIR}/data/nq_data/dev.jsonl ${DATA_DIR}/data/nq_data/test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-nq
PRECISION="fp32" # "bf16"
if [[ "$size" == "xxl" ]] || [[ "$size" == "xl" ]] ; then
    TOTAL_STEPS=5000
else
    TOTAL_STEPS=10000
fi

srun python train.py \
    --shuffle \
    --train_retriever --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever\
    --precision ${PRECISION} \
    --shard_optim --shard_grads \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index 0-1000:500,1000-10000:2000 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 4e-5 --lr_retriever 4e-5 \
    --scheduler cosine \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 40 --retriever_n_context 40 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 500 \
    --log_freq 50 \
    --total_steps ${TOTAL_STEPS} \
    --warmup_steps 100 \
    --save_freq 5000 \
    --main_port $port \
    --write_results \
    --task qa \
    --index_mode flat \
    --load_index_path ${PRETRAINED_INDEX}
