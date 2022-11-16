#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --job-name=nq
#SBATCH --output=run_dir/%A.out
#SBATCH --error=run_dir/%A.err
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --mem=470GB
#SBATCH --partition=devlab
#SBATCH --constraint=volta32gb

# Example of running retrieval from an example corpus (Wikipedia 2018 - 30M passsages) using Atlas' stand alone retriever mode: 
# First, we'll download the resources we need, then embed the corpus, save the index to disk, then run retrieval over some QA pairs, and save retrieval results

size=xl
DATA_DIR='/checkpoint/plewis/atlas_opensourcing_check/'
port=$(shuf -i 15000-16000 -n 1)

# download the NQ data:
# python preprocessing/prepare_nq.py --output_directory ${DATA_DIR} 

# download the Wikipedia 2018 corpus:
# python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR}

# downloads pretrained ATLAS-large:
# python preprocessing/download_model.py --model models/atlas_nq/${size} --output_directory ${DATA_DIR}

# we'll retrieve from the following passages:
PASSAGES_TO_RETRIEVE_FROM="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl ${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"

# run retrieval for the Natural Questions dev and test data
EVAL_FILES="${DATA_DIR}/data/nq_data/dev.jsonl ${DATA_DIR}/data/nq_data/test.jsonl" # run retreival for the Natural Questions dev and test data

# we'll retrieve using the ATLAS pretrained retriever, subsequently finetuned one Natural Questions
PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
# or, uncomment the next line to use standard contriever weights for retrieval instead:
# PRETRAINED_MODEL=none

SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${size}-nq-retrieve-only


srun python evaluate.py \
    --name ${EXPERIMENT_NAME} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${PRETRAINED_MODEL} \
    --eval_data ${EVAL_FILES} \
    --n_context 40 --retriever_n_context 40 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat" \
    --task "qa" \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index \
    --write_results \
    --retrieve_only \
    --passages ${PASSAGES_TO_RETRIEVE_FROM}

# observe the logs at ${SAVE_DIR}/${EXPERIMENT_NAME}/run.log. 
# Retrieval results will be saved in ${SAVE_DIR}/${EXPERIMENT_NAME}, and the retrieval index will be saved to ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index
