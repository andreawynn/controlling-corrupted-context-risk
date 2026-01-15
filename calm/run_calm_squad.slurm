#!/bin/bash
#SBATCH --job-name=calm_squad         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem=75G                # memory per node
#SBATCH --array=0-1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_output/%j.out

CONTEXT_CONDITION='u' # 'a' or 'u' - answerable or unanswerable
RUN_ZEROSHOT='N' # 'Y' or 'N'; will either run ONLY with context (all lambdas) or ONLY zeroshot (one lambda)
LAMBDA=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" lams.txt)

echo "Running with LAMBDA=$LAMBDA and CONTEXT_CONDITION=$CONTEXT_CONDITION and RUN_ZEROSHOT=$RUN_ZEROSHOT"

python src/run_question_answering.py --model_name_or_path google-t5/t5-large --do_eval --use_data_subset True --dataset_name squad_v2 --context_column context --question_column question --answer_column answers --output_dir ./save/squad_t5-large/ --per_device_eval_batch_size 1 --deploy_scenario True --use_synchronize True --overwrite_output_dir --predict_with_generate --max_seq_length 512 --use_early_exit True --exit_conf_type softmax --exit_conf_threshold "$LAMBDA" --exit_min_layer 12 --include_inputs_for_metrics False --use_auth_token False --type_vocab_reduct True --k 256 --context_condition "$CONTEXT_CONDITION" --run_zeroshot "$RUN_ZEROSHOT"