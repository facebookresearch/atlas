SIZE=base
DATA_DIR='/data/side/gnr/runs_marialomeli/atlas_oss'
TRAIN_FILE="${DATA_DIR}/nq_data/train.64-shot.jsonl"
EVAL_FILES="${DATA_DIR}/nq_data/dev.jsonl ${DATA_DIR}/nq_data/test.jsonl"
PORT=$(shuf -i 15000-16000 -n 1)
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${SIZE}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${SIZE}
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${SIZE}-nq-64-shot
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${SIZE}

python finetune_qa.py  --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --name $EXPERIMENT_NAME \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 5 \
    --log_freq 4 \
    --total_steps 10 \
    --warmup_steps 5 \
    --save_freq 9 \
    --main_port $PORT \
    --write_results \
    --index_mode flat \
   --model_path $PRETRAINED_MODEL \
   --reader_model_type  "google/t5-${SIZE}-lm-adapt" \
    --load_index_path $PRETRAINED_INDEX \