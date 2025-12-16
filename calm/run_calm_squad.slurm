#!/bin/bash
#SBATCH --job-name=calm_squad         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --mem-per-cpu=150G         # memory per cpu-core
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./slurm_output/%j.out

python src/run_question_answering.py --model_name_or_path google-t5/t5-large --do_eval --dataset_name squad --context_column context --question_column question --answer_column answers --output_dir ./save/squad_t5-large/ --per_device_eval_batch_size 1 --deploy_scenario True --use_synchronize True --overwrite_output_dir --predict_with_generate --max_seq_length 512 --use_early_exit True --exit_conf_type softmax --exit_conf_threshold 0.9 --exit_min_layer 19 --include_inputs_for_metrics False --use_auth_token True --type_vocab_reduct True --k 256