#!/usr/bin/env python
# coding=utf-8

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import pandas as pd
import logging
import os
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
#     Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import IntervalStrategy


from arguments import ModelArguments, DataTrainingArguments, summarization_name_mapping
from dynamic_partition_model import T5ForConditionalGeneration_updated
from trainer import Seq2SeqTrainer_updated
from generation import GenerationMixin_updated
from dataloader import DataCollatorForSeq2Seq


logger = logging.getLogger(__name__)

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
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

if data_args.source_prefix is None and model_args.model_name_or_path in [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
]:
    logger.warning(
        "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
        "`--source_prefix 'summarize: ' `"
    )

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


# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files this script will use the first column for the full texts and the second column for the
# summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    if data_args.dataset_name == 'cnn_dailymail':
        raw_datasets = load_dataset(
           data_args.dataset_name, '3.0.0', data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        raw_datasets = load_dataset(
           data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
else:
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["test"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = T5ForConditionalGeneration_updated.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    n_facet_all = model_args.n_facet_all, 
    n_facet=model_args.n_facet, 
    n_facet_context =model_args.n_facet_context, 
    n_facet_reranker =model_args.n_facet_reranker,
    n_facet_local_encoder=model_args.n_facet_local_encoder,
    n_facet_hidden=model_args.n_facet_hidden, 
    n_facet_window=model_args.n_facet_window, 
    n_facet_MLP=model_args.n_facet_MLP, 
    use_proj_bias=False, 
    weight_mode = '', 
    softmax_nonlinear='None', 
    context_efficient_mode='assign_average_of_duplicates', 
    reranker_efficient_mode='assign_only_last_of_duplicates', 
    context_source=model_args.context_source,
    reranker_CAN_NUM=model_args.reranker_CAN_NUM, 
    pointer_mode=model_args.pointer_mode,
    candidates_from_previous_reranker=True
)

def init_project_arr_for_t5(model):        
    if model.n_facet_local_encoder > 0:
        for i in range(model.n_facet_local_encoder):
            model.encoder_project_arr[i].weight.data = 1e-10 * torch.eye(model.model_dim)
    return
     
if training_args.do_train:
    init_project_arr_for_t5(model)

model.resize_token_embeddings(len(tokenizer))

prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

# Preprocessing the datasets.
# We need to tokenize inputs and targets.
if training_args.do_train:
    column_names = raw_datasets["train"].column_names
elif training_args.do_eval:
    column_names = raw_datasets["validation"].column_names
elif training_args.do_predict:
    column_names = raw_datasets["test"].column_names
else:
    logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")

# Get the column names for input/target.
dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
if data_args.text_column is None:
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
else:
    text_column = data_args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
        )
if data_args.summary_column is None:
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
else:
    summary_column = data_args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )

# Temporarily set max_target_length for training.
max_target_length = data_args.max_target_length
padding = "max_length" if data_args.pad_to_max_length else False

def preprocess_function(examples):
    # remove pairs where at least one record is None
    # set_trace()
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] is not None and examples[summary_column][i] is not None:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=data_args.max_target_length, padding="max_length", truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


#define preproc_data_dir in DataTrainingArguments

from datasets import load_dataset, load_metric, load_from_disk

if training_args.do_train:
    file_path = data_args.preproc_data_dir + '/' + data_args.dataset_name + '_train'
    if os.path.exists(file_path):
        train_dataset = load_from_disk(file_path)
    else:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        train_dataset.save_to_disk(file_path)
        
if training_args.do_eval:
    file_path = data_args.preproc_data_dir + '/' + data_args.dataset_name + '_validation'
    if os.path.exists(file_path):
        eval_dataset = load_from_disk(file_path)
    else:
        if "validation" not in raw_datasets:
            raise ValueError("--do_validation requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        eval_dataset.save_to_disk(file_path)
        
if training_args.do_predict:
    file_path = data_args.preproc_data_dir + '/' + data_args.dataset_name + '_test'
    if os.path.exists(file_path):
        test_dataset = load_from_disk(file_path)
    else:
        if "test" not in raw_datasets:
            raise ValueError("--do_test requires a train dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        test_dataset.save_to_disk(file_path)
    
# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)


# Metric
import evaluate
metric_rouge = evaluate.load("rouge")
metric_meteor = evaluate.load("meteor")
# metric_mauve = evaluate.load("mauve")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

from nltk.translate import nist_score 
from nltk import word_tokenize
import spacy
from cider import Cider
nlp = spacy.load("en_core_web_sm")

def compute_cider(predictions, references):
    hypotheses = []
    for text in predictions:
        hypotheses.append(' '.join(word_tokenize(text)))

    list_of_references = []
    for text in references:
        list_of_references.append([' '.join(word_tokenize(text))])

    metric_cider = Cider()
    return  metric_cider.compute_score(list_of_references, hypotheses)[0]

def compute_nist(predictions, references):
    hypotheses = []
    for text in predictions:
        hypotheses.append(word_tokenize(text))

    list_of_references = []
    for text in references:
        list_of_references.append([word_tokenize(text)])

    return nist_score.corpus_nist(list_of_references, hypotheses, n=5)

def only_proper(input_text):
    output_text = []
    for text in input_text:
        output_text.append(' '.join( [tok.text for tok in nlp(text) if tok.pos_ == "PROPN"] ) )
    return output_text

def postprocess_text(preds, labels, inputs):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    inputs = [input_.strip() for input_ in inputs]

#     # rougeLSum expects newline after each sentence
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels, inputs

global model_gen
model_gen = {}

def compute_metrics(eval_preds):
    preds, labels, inputs = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels, decoded_inputs = postprocess_text(decoded_preds, decoded_labels, decoded_inputs)

    result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v * 100, 4) for k, v in result.items()}
    
    result_proper = metric_rouge.compute(predictions=only_proper(decoded_preds), references=only_proper(decoded_labels), rouge_types=['rouge1'])
    result['rouge1_proper'] = result_proper['rouge1']
    
    #result_mauve = metric_mauve.compute(predictions=decoded_preds, references=decoded_labels)
    #result['mauve'] = result_mauve.mauve
    result['Cider'] = compute_cider(predictions=decoded_preds, references=decoded_labels)
    result['NIST'] = compute_nist(predictions=decoded_preds, references=decoded_labels)
    result_meteor = metric_meteor.compute(predictions=decoded_preds, references=decoded_labels)
    result.update(result_meteor)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    
    global model_gen
    model_gen['decoded_preds'] = decoded_preds
    model_gen['decoded_labels'] = decoded_labels
    model_gen['decoded_inputs'] = decoded_inputs
    
    return result


# Initialize our Trainer
training_args.num_train_epochs = data_args.no_train_epochs
training_args.warmup_steps = 1000

training_args.logging_strategy = IntervalStrategy.EPOCH
# training_args.logging_strategy = IntervalStrategy.STEPS
# training_args.logging_steps = 1000

training_args.save_strategy =  IntervalStrategy.NO
# training_args.save_total_limit = 3

training_args.evaluation_strategy = IntervalStrategy.NO
# training_args.evaluation_strategy = IntervalStrategy.STEPS
# training_args.eval_steps = 10000 #5000


print(training_args.__dict__)
# assert 0
callbacks = []

trainer = Seq2SeqTrainer_updated(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    callbacks=callbacks,
)

def print_parameters(model, excluding_prefix_arr = None):
    parameter_sum = 0
    for name, param in model.named_parameters():
        contiue_or_not = False
        for excluding_prefix in excluding_prefix_arr:
            if excluding_prefix is not None and excluding_prefix == name[:len(excluding_prefix)]:
                print("Skipping ", name, param.numel())
                contiue_or_not = True
                break
        if contiue_or_not:
            continue
        if param.requires_grad:
            print(name, param.numel())
            parameter_sum += param.numel()
    return parameter_sum

skip_parameters = []
if model_args.n_facet == 1:
    skip_parameters+= ["weight_global",'weight_facet_decoder']

total_params = sum(x.data.nelement() for x in model.parameters())
print('total parameters: ', total_params)
print('total parameters (exclude duplication):', (print_parameters(model, skip_parameters)))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

# set_trace()
# Training
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
if training_args.do_eval:
    eval_results = trainer.evaluate()
    print("eval_results", eval_results)

    model_gen_df = pd.DataFrame(model_gen)
    model_gen_df.to_csv(training_args.output_dir+'/model_generation.csv')
