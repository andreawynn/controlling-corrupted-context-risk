#!/bin/bash
#SBATCH --job-name=adv_experiment         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=250G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:3             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./slurm_output/%j.out

export HUGGINGFACE_TOKEN="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"
# Also try some instruct models, if I have space
# meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.1-8B-Instruct

# 0: ALERT
export DATASET_INDEX=0

# Specify the number of candidate responses for the model to generate
export NUM_CANDIDATE_RESPONSES=15

# Whether to run adversarial prompts (Y/N)
# no means run on non-adversarial prompts from the same dataset
export USE_ADVERSARIAL_PROMPTS="Y"

# Set to Y if using early exit, else sample multiple responses from the model
export EARLY_EXIT="Y"

# 0 and 2: LLaMA 3 8B models
# 1 and 3: LLaMA 2 7B models
# 4: LLaMA 2 13B model
# 'a' = all models
export MODEL_INDEX='a'

export RESULT_FOLDER_NAME="results_adversarial"

python adversarial_experiment.py