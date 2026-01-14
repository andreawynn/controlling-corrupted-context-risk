#!/bin/bash
#SBATCH --job-name=compute_calibration         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --mem-per-cpu=250G         # memory per cpu-core
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./slurm_output/%j.out

export HUGGINGFACE_TOKEN="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"
export N_DEMOS=36

# 0 through 6
# 0 and 2: LLaMA 3 8B models
# 1 and 3: LLaMA 2 7B models
# 4: LLaMA 2 13B model
# 'a': run all at once
export MODEL_INDEX='a'

export USE_FAKE_LABELS='Y'

# datasets = [0'trec', 1'ag_news', 2'financial_phrasebank', 3'sst2', 4'tweeteval_hate', 5'tweeteval_feminist', 
#            6'tweeteval_atheism', 7'unnatural']
# 'a': run all at once
export DATASET_INDEX=7
python compute_calibration_matrices.py