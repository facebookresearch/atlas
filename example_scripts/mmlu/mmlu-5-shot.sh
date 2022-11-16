#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --job-name=mmlu
#SBATCH --output=run_dir/%A.out
#SBATCH --error=run_dir/%A.err
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --array=0-56%4

size=xl
DATA_DIR='/checkpoint/plewis/atlas_opensourcing/data/'

DOMAINS=(
    "abstract_algebra" \
    "anatomy" \
    "astronomy" \
    "business_ethics" \
    "clinical_knowledge" \
    "college_biology" \
    "college_chemistry" \
    "college_computer_science" \
    "college_mathematics" \
    "college_medicine" \
    "college_physics" \
    "computer_security" \
    "conceptual_physics" \
    "econometrics" \
    "electrical_engineering" \
    "elementary_mathematics" \
    "formal_logic" \
    "global_facts" \
    "high_school_biology" \
    "high_school_chemistry" \
    "high_school_computer_science" \
    "high_school_european_history" \
    "high_school_geography" \
    "high_school_government_and_politics" \
    "high_school_macroeconomics" \
    "high_school_mathematics" \
    "high_school_microeconomics" \
    "high_school_physics" \
    "high_school_psychology" \
    "high_school_statistics" \
    "high_school_us_history" \
    "high_school_world_history" \
    "human_aging" \
    "human_sexuality" \
    "international_law" \
    "jurisprudence" \
    "logical_fallacies" \
    "machine_learning" \
    "management" \
    "marketing" \
    "medical_genetics" \
    "miscellaneous" \
    "moral_disputes" \
    "moral_scenarios" \
    "nutrition" \
    "philosophy" \
    "prehistory" \
    "professional_accounting" \
    "professional_law" \
    "professional_medicine" \
    "professional_psychology" \
    "public_relations" \
    "security_studies" \
    "sociology" \
    "us_foreign_policy" \
    "virology" \
    "world_religions" \
)
DOMAIN=${DOMAINS[${SLURM_ARRAY_TASK_ID}]}
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/data/mmlu_data/5-shot/individual_train/${DOMAIN}.5-shot-train.jsonl"
EVAL_FILES="${DATA_DIR}/data/mmlu_data/5-shot/individual_valid/${DOMAIN}.val.jsonl ${DATA_DIR}/data/mmlu_data/5-shot/individual_valid/${DOMAIN}.test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
SAVE_DIR=${DATA_DIR}/experiments/${SLURM_ARRAY_JOB_ID}-${size}-mmlu-5-shot/
EXPERIMENT_NAME=$DOMAIN
PRECISION="fp32" # "bf16"

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
