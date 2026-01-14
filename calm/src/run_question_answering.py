#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.


TESTING = True

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import logging
import os
import sys
import nltk
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score
from functools import partial

import seaborn as sns
import pandas as pd

import copy
import datasets
import evaluate
import transformers
from datasets import load_dataset    
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import T5ForConditionalGeneration as HfT5ForConditionalGeneration
from transformers import Seq2SeqTrainer

from qa_lib import (
    ModelArguments,
    DataTrainingArguments,
    QATrainer,
    adjust_training_args,
)
from models import (
    T5ForConditionalGeneration,
    DeployT5ForConditionalGeneration,
)
from util import (
    AdditionalArguments,
    update_autoconfig,
)
import wandb

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

question_answering_column_name_mapping = {
    "squad": ("question", "context", "answer"),
    "squad_v2": ("question", "context", "answer"),
    "narrativeqa": ("question", "document", "answer")
}

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main(model_args, data_args, training_args, additional_args, model_cls, trainer_cls, jupyter=False):    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_seq2seq_qa", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    model_args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    # Separate answerable vs un-answerable questions for SQuAD2.0
    if data_args.dataset_name == "squad_v2":
        # Define a helper to check answerability
        def is_answerable(example):
            # The 'text' list is empty for unanswerable examples
            return len(example["answers"]["text"]) > 0

        # Filter the splits
        train_answerable = raw_datasets["train"].filter(is_answerable)
        train_unanswerable = raw_datasets["train"].filter(lambda ex: not is_answerable(ex))

        val_answerable = raw_datasets["validation"].filter(is_answerable)
        val_unanswerable = raw_datasets["validation"].filter(lambda ex: not is_answerable(ex))

        # Only do one of 'answerable' or 'unanswerable'
        if data_args.context_condition == 'a':
            # Use answerable context
            raw_datasets['train'] = train_answerable
            raw_datasets['validation'] = val_answerable
            wandb.log({"context_condition": "answerable"})
        elif data_args.context_condition == 'u':
            # Use unanswerable context
            raw_datasets['train'] = train_unanswerable
            raw_datasets['validation'] = val_unanswerable
            wandb.log({"context_condition": "unanswerable"})
        else:
            raise NotImplementedError("Context condition not recognized. Must be answerable (a) or unanswerable (u).")

    # For debugging: use a small subset of data
    if additional_args.use_data_subset:
        raw_datasets['train'] = raw_datasets['train'].select(range(50))
        raw_datasets['validation'] = raw_datasets['validation'].select(range(50))
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if not additional_args.use_lora or training_args.do_train:
        config_name = model_args.config_name if model_args.config_name else model_args.model_name_or_path
        tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        model_name = model_args.model_name_or_path
    else:
        lora_config = LoraConfig.from_pretrained(model_args.model_name_or_path)
        config_name = tokenizer_name = model_name = lora_config.base_model_name_or_path
        
    config = AutoConfig.from_pretrained(
        config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config = update_autoconfig(
        config, 
        additional_args,
        max_answer_length=data_args.max_answer_length,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    model = model_cls.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
        
    if additional_args.use_lora:
        if training_args.do_train:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, r=additional_args.lora_rank, 
                lora_alpha=additional_args.lora_alpha, lora_dropout=additional_args.lora_dropout,
                target_modules=additional_args.lora_target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, config=lora_config)
            model = model.merge_and_unload()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = question_answering_column_name_mapping.get(data_args.dataset_name, None)
    if data_args.question_column is None:
        question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        question_column = data_args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.context_column is None:
        context_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        context_column = data_args.context_column
        if context_column not in column_names:
            raise ValueError(
                f"--context_column' value '{data_args.context_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.answer_column is None:
        answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        answer_column = data_args.answer_column
        if answer_column not in column_names:
            raise ValueError(
                f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_answer_length for training.
    max_answer_length = data_args.max_answer_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_squad_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
        zeroshot: bool = False,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        contexts = examples[context_column]
        answers = examples[answer_column]

        def generate_input(_question, _context):
            if zeroshot:
                return " ".join(["question:", _question.lstrip()])
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        return inputs, targets
    
    def preprocess_narrativeqa_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        contexts = [e['summary']['text'] for e in examples[context_column]]
        answers = examples[answer_column]
        
        def generate_input(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        inputs = [generate_input(question['text'], context) for question, context in zip(questions, contexts)]
        targets = [answer[0]["text"] if len(answer) > 0 else "" for answer in answers]
        return inputs, targets

    def preprocess_function(examples, zeroshot: bool = False):
        if "squad" in data_args.dataset_name:
            inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column, zeroshot)
        elif data_args.dataset_name == "narrativeqa":
            inputs, targets = preprocess_narrativeqa_batch(examples, question_column, context_column, answer_column)
        else:
            raise NotImplementedError
            
        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Tokenize targets with text_target=...
        labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Validation preprocessing
    def preprocess_validation_function(examples, zeroshot: bool = False):            
        if "squad" in data_args.dataset_name:
            inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column, zeroshot)
            model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True,
                                     return_overflowing_tokens=True, return_offsets_mapping=True,)

        elif data_args.dataset_name == "narrativeqa":
            inputs, targets = preprocess_narrativeqa_batch(examples, question_column, context_column, answer_column)
            model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
            
        else:
            raise NotImplementedError

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        if "squad" in data_args.dataset_name:
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            model_inputs["example_id"] = []
            # Augment the overflowing tokens to the labels
            labels_out = []
            for i in range(len(model_inputs["input_ids"])):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                model_inputs["example_id"].append(examples["id"][sample_index])
                labels_out.append(labels["input_ids"][sample_index])

            model_inputs["labels"] = labels_out
            
        elif data_args.dataset_name == "narrativeqa":
            model_inputs["labels"] = labels["input_ids"]
            
        else:
            raise NotImplementedError
            
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        preprocess_fn = partial(preprocess_function, zeroshot=False)
        with training_args.main_process_first(desc="train dataset map pre-processing (with context)"):
            train_dataset = train_dataset.map(
                preprocess_fn, 
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset (with context)",
            )
        # Create zero-shot train feature from dataset
        zs_preprocess_fn = partial(preprocess_function, zeroshot=True)
        with training_args.main_process_first(desc="train dataset map pre-processing (without context)"):
            zeroshot_dataset = train_dataset.map(
                zs_preprocess_fn, 
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset (without context)",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            zeroshot_dataset = zeroshot_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        preprocess_fn = partial(preprocess_validation_function, zeroshot=False)
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        # Zero-Shot Validation Feature Creation
        zs_preprocess_fn = partial(preprocess_validation_function, zeroshot=True)
        with training_args.main_process_first(desc="validation dataset map pre-processing (zeroshot)"):
            zs_eval_dataset = eval_examples.map(
                zs_preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset (zeroshot)",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
            zs_eval_dataset = zs_eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        preprocess_fn = partial(preprocess_validation_function, zeroshot=False)
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        # Zero-Shot Predict Feature Creation
        zs_preprocess_fn = partial(preprocess_validation_function, zeroshot=True)
        with training_args.main_process_first(desc="prediction dataset map pre-processing (zeroshot)"):
            zs_predict_dataset = predict_examples.map(
                zs_preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset (zeroshot)",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(zs_predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
            zs_predict_dataset = zs_predict_dataset.select(range(max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    if data_args.dataset_name == "narrativeqa":
        metric = evaluate.load("rouge")
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels
        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            try:
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            except:
                decoded_preds = tokenizer.batch_decode(np.where(preds != -100, preds, tokenizer.pad_token_id), 
                                                       skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            try:
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            except:
                decoded_labels = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), 
                                                        skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result
        
    elif "squad" in data_args.dataset_name:
        metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
        
        def compute_metrics(p: EvalPrediction, prefix: str = None, compute_losses: bool = True):   
            # Check if all references have empty text lists (unanswerable questions)
            all_empty = all(
                len(ref.get("answers", {}).get("text", [])) == 0 
                for ref in p.label_ids
            )
            
            if all_empty and len(p.label_ids) > 0:
                # Handle the case where all questions are unanswerable
                # Return default metrics for unanswerable questions
                metric_dict = {
                    "exact_match": 0.0,
                    "f1": 0.0,
                }
                if data_args.version_2_with_negative:
                    metric_dict["HasAns_exact"] = 0.0
                    metric_dict["HasAns_f1"] = 0.0
                    metric_dict["NoAns_exact"] = 0.0
                    metric_dict["NoAns_f1"] = 0.0
                    metric_dict["best_exact"] = 0.0
                    metric_dict["best_f1"] = 0.0
                    metric_dict["total"] = len(p.label_ids)
                    metric_dict["HasAns_total"] = 0
                    metric_dict["NoAns_total"] = len(p.label_ids)
            else:
                try:
                    metric_dict = metric.compute(predictions=p.predictions, references=p.label_ids)
                except ValueError as e:
                    # Handle the case where metric_max_over_ground_truths fails due to empty sequences
                    if "max() arg is an empty sequence" in str(e):
                        # Filter out examples with empty ground truths and compute metrics on the rest
                        filtered_predictions = []
                        filtered_references = []
                        for pred, ref in zip(p.predictions, p.label_ids):
                            if len(ref.get("answers", {}).get("text", [])) > 0:
                                filtered_predictions.append(pred)
                                filtered_references.append(ref)
                        
                        if len(filtered_predictions) > 0:
                            metric_dict = metric.compute(predictions=filtered_predictions, references=filtered_references)
                        else:
                            # All examples have empty ground truths
                            metric_dict = {
                                "exact_match": 0.0,
                                "f1": 0.0,
                            }
                            if data_args.version_2_with_negative:
                                metric_dict["HasAns_exact"] = 0.0
                                metric_dict["HasAns_f1"] = 0.0
                                metric_dict["NoAns_exact"] = 0.0
                                metric_dict["NoAns_f1"] = 0.0
                                metric_dict["best_exact"] = 0.0
                                metric_dict["best_f1"] = 0.0
                                metric_dict["total"] = len(p.label_ids)
                                metric_dict["HasAns_total"] = 0
                                metric_dict["NoAns_total"] = len(p.label_ids)
                    else:
                        raise e
            
            metric_keys = deepcopy(list(metric_dict.keys()))
            for key in metric_keys:
                if prefix is not None and prefix not in key:
                    metric_dict['{}_{}'.format(prefix, key)] = metric_dict.pop(key)
            if compute_losses:
                losses = []
                for i in range(len(p.predictions)):
                    try:
                        loss_metric = metric.compute(predictions=[p.predictions[i]], references=[p.label_ids[i]])
                        losses.append(loss_metric.get('f1', 0.0))
                    except (ValueError, KeyError):
                        # Handle empty ground truths for individual examples
                        losses.append(0.0)
                metric_dict['losses'] = str(losses)
            return metric_dict
        
    else:
        raise NotImplementedError
        
    # Post-processing:
    def post_processing_function(
        examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}

        task_name = "squad_v2" if data_args.version_2_with_negative else "squad"

        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            try:
                feature_index = feature_per_example[example_index]
                predictions[example["id"]] = decoded_preds[feature_index]
            except:
                predictions[example["id"]] = "fail"

        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # adjust training arguments
    training_args = adjust_training_args(training_args, data_args, additional_args)

    def train_eval_predict(trainer, train_dataset, predict_dataset, eval_dataset, zeroshot: bool):
        zeroshot_str = "_zeroshot" if zeroshot else ""
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            if not additional_args.use_lora: trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train" + zeroshot_str, metrics)
            trainer.save_metrics("train" + zeroshot_str, metrics)
            trainer.save_state()

            # Log to wandb summary; this way we also preserve zeroshot
            wandb.run.summary[f"train{zeroshot_str}/train_samples"] = metrics["train_samples"]
            wandb.run.summary[f"train{zeroshot_str}/train_exact_match"] = metrics["train_exact_match"]
            if zeroshot:
                # Used the full model
                wandb.run.summary[f"train{zeroshot_str}/avg_exit"] = model.config.num_hidden_layers
            else:
                wandb.run.summary[f"train{zeroshot_str}/avg_exit"] = metrics["train_block_avg"]
            
            if additional_args.use_lora:
                model.save_pretrained(training_args.output_dir)  # save adapter_config.json
                model.base_model.save_pretrained(training_args.output_dir)  # save config.json

        # Evaluation
        results = {}
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_answer_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            # evaluation metrics could be differ from evaluation during training
            # refer to https://discuss.huggingface.co/t/evaluation-results-metric-during-training-is-different-from-the-evaluation-results-at-the-end/15401/3
            if training_args.include_inputs_for_metrics and not zeroshot:
                output = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
                metrics = output.metrics
                if additional_args.count_flops:
                    final_flops = model.decoder.flop_counter/model.decoder.count_passes
                    wandb.log({"FLOP/Token" + zeroshot_str: final_flops})
                    flops_sample = model.decoder.flop_counter/len(eval_dataset)
                    wandb.log({"FLOP/Sample" + zeroshot_str: flops_sample})
                    total_flops = model.decoder.flop_counter
                    wandb.log({"Total_FLOP" + zeroshot_str: total_flops})
            else:
                metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

            # Convert to a mutable dictionary
            if isinstance(eval_output, dict):
                metrics = eval_output
            else:
                # It's an EvalLoopOutput object, extract the metrics
                metrics = eval_output.metrics if hasattr(eval_output, 'metrics') else {}

            if additional_args.plotting_logits:
                data = model.decoder.graph_top_k_list
                max_length = max(len(arr) for arr in data)

                # Pad arrays with NaNs to ensure they are all the same length
                padded_data = [np.pad(np.array(arr, dtype=float),  # Convert array to float
                            (0, max_length - len(arr)),
                            mode='constant',
                            constant_values=np.nan)
                    for arr in data]
                
                # Convert the list of arrays into a single NumPy array
                padded_array = np.array(padded_data)

                # Converting the array to a DataFrame for easier handling in seaborn
                df = pd.DataFrame(padded_array)
                table = wandb.Table(dataframe=df)
                wandb.log({'plotting_logits' + zeroshot_str: table})

                # Creating a boxplot
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=df)
                plt.title('Rank of the final predicted token at each layer', fontsize=20)
                plt.xlabel('Layer', fontsize=16)
                plt.ylabel('Rank of the final predicted token', fontsize=16)
                plt.grid(True)
                file_path = "plots/boxplot_topk_rank_eval" + data_args.dataset_name.replace("/", "_") + "_" + model_args.model_name_or_path.replace("/", "_") + ".png"
                plt.savefig(file_path)
                wandb.log({"Boxplot" + zeroshot_str: wandb.Image(file_path)})

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval" + zeroshot_str, metrics)
            trainer.save_metrics("eval" + zeroshot_str, metrics)

            # Log to wandb summary; this way we also preserve zeroshot
            wandb.run.summary[f"eval{zeroshot_str}/eval_samples"] = metrics["eval_samples"]
            wandb.run.summary[f"eval{zeroshot_str}/eval_exact_match"] = metrics["eval_exact_match"]
            if zeroshot:
                # Used the full model
                wandb.run.summary[f"eval{zeroshot_str}/avg_exit"] = model.config.num_hidden_layers
            else:
                wandb.run.summary[f"eval{zeroshot_str}/avg_exit"] = metrics["eval_block_avg"]

        # Prediction
        if training_args.do_predict:
            logger.info("*** Predict ***")
            results = trainer.predict(predict_dataset, predict_examples)
            metrics = results.metrics

            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            trainer.log_metrics("predict" + zeroshot_str, metrics)
            trainer.save_metrics("predict" + zeroshot_str, metrics)

            # Log to wandb summary; this way we also preserve zeroshot
            wandb.run.summary[f"predict{zeroshot_str}/predict_samples"] = metrics["predict_samples"]
            wandb.run.summary[f"predict{zeroshot_str}/predict_exact_match"] = metrics["predict_exact_match"]
            if zeroshot:
                # Used the full model
                wandb.run.summary[f"predict{zeroshot_str}/avg_exit"] = model.config.num_hidden_layers
            else:
                wandb.run.summary[f"predict{zeroshot_str}/avg_exit"] = metrics["predict_block_avg"]
            

        if training_args.push_to_hub:
            kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering" + zeroshot_str}
            if data_args.dataset_name is not None:
                kwargs["dataset_tags"] = data_args.dataset_name
                if data_args.dataset_config_name is not None:
                    kwargs["dataset_args"] = data_args.dataset_config_name
                    kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
                else:
                    kwargs["dataset"] = data_args.dataset_name

            trainer.push_to_hub(**kwargs)

        return results, metrics
        
    if data_args.run_zeroshot == 'N':
        train_data = train_dataset if training_args.do_train else None
        eval_data = eval_dataset if training_args.do_eval else None
        eval_ex = eval_examples if training_args.do_eval else None
        predict_data = predict_dataset if training_args.do_predict else None
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            eval_examples=eval_ex,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics, # if training_args.predict_with_generate else None,
            post_process_function=post_processing_function if "squad" in data_args.dataset_name else None,
        )

        results, metrics = train_eval_predict(trainer, train_data, predict_data, eval_data, zeroshot=False)
    elif data_args.run_zeroshot == 'Y':
        zs_train_data = zs_train_dataset if training_args.do_train else None
        zs_eval_data = zs_eval_dataset if training_args.do_eval else None
        zs_predict_data = zs_predict_dataset if training_args.do_predict else None
        # For zero-shot use the same CALM model but with zero-shot datasets and lambda=1.0
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=zs_train_data,
            eval_dataset=zs_eval_data,
            eval_examples=eval_ex,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics, # if training_args.predict_with_generate else None,
            post_process_function=post_processing_function if "squad" in data_args.dataset_name else None,
        )
        results, metrics = train_eval_predict(trainer, zs_train_data, zs_predict_data, zs_eval_data, zeroshot=True)
    else:
        raise ValueError("data_args.run_zeroshot must be 'Y' or 'N'")

    if not jupyter:
        return results, metrics
    else:
        return trainer, metrics


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Initialize a new run
    wandb.login()
    wandb.init(
        project="calm-squad",
        entity="awynn13-johns-hopkins-university",
        mode="online",
        # track hyperparameters and run metadata
        config={
            "dataset": data_args.dataset_name,
            "model": model_args.model_name_or_path, 
            "exit_conf_type": additional_args.exit_conf_type,
            "exit_conf_threshold": additional_args.exit_conf_threshold,
            "exit_min_layer": additional_args.exit_min_layer,
            "type_vocab_reduct": additional_args.type_vocab_reduct,
            "lambda_threshold": additional_args.exit_conf_threshold,
            },
    )

    # Make sure if we're running the zero-shot model that lambda=1.0
    if data_args.run_zeroshot == 'Y':
        additional_args.exit_conf_threshold = 1.0

    if data_args.dataset_name in ["squad", "squad_v2", "narrativeqa"]:
        model_cls = T5ForConditionalGeneration if not additional_args.deploy_scenario else DeployT5ForConditionalGeneration
    trainer_cls = QATrainer
    training_args.include_inputs_for_metrics = True

    results, metrics = main(model_args, data_args, training_args, additional_args, model_cls, trainer_cls)

    # Save out the results to a local file
    folder = "outputs/lambda_" + str(additional_args.exit_conf_threshold) + "/"
    os.makedirs(folder, exist_ok=True)
    zeroshot_str = "" if data_args.run_zeroshot == 'N' else "_zeroshot"
    with open(folder + data_args.context_condition + zeroshot_str + "_losses.txt", "w") as file:
        file.write(metrics.get("losses", ""))
        # Also write out the average exit on a new line
        file.write("\n")
        file.write(wandb.run.summary[f"eval{zeroshot_str}/avg_exit"])

    try:
        wandb.finish()
    except:
        pass
