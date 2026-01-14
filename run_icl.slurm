#!/bin/bash
#SBATCH --job-name=icl_experiment         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=250G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./slurm_output/%j.out

export HUGGINGFACE_TOKEN="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"
# Also try some instruct models, if I have space
# meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.1-8B-Instruct

# datasets = [0'trec', 1'ag_news', 2'financial_phrasebank', 3'sst2', 4'tweeteval_hate', 5'tweeteval_feminist', 
#            6'tweeteval_atheism', 7'unnatural']
export DATASET_INDEX=0
export N_DEMOS=36
export USE_CALIBRATION='Y' # Y or N
export USE_FAKE_LABELS='Y'

# 0 and 2: LLaMA 3 8B models
# 1 and 3: LLaMA 2 7B models
# 4: LLaMA 2 13B model
# 'a' = all models
export MODEL_INDEX='a'

export RESULT_FOLDER_NAME="results_fake_labels"

# c = correct, i = incorrect, z = zeroshot, a = all 
# If type = a, increase runtime to 12hr. Ideal for overnight and weekend jobs, not quick turnaround tests. 
export EXPERIMENT_TYPE="a"

python icl_experiment.py