#---------------------------------------------
# DeepFrog
#   by Maarten van Gompel
#   Centre for Language and Speech Technology, Radboud University, Nijmegen
#---------------------------------------------
#Licensed under GNU General Public License v3
#---------------------------------------------

# Significant parts of this code are derived and adapted from run_ner.py and ner_utils.py in Huggingface's Transformers,
# which was licensed as follows:
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.  Copyright (c) 2018, NVIDIA
# CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations under the License.


import sys
import os
import argparse
import glob
import logging
import random
import shutil

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from deepfrog.data import TaggerInputDataset #convert_examples_to_features, get_labels, read_examples_from_file


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            super(AttrDict, self).__getattr__(key)

    def __setattr__(self, key, value):
        if key[0] != '_':
            self[key] = value
        else:
            super(AttrDict, self).__setattr__(key, value)


class Tagger:
    def __init__(self, **kwargs):
        self.args = AttrDict()
        self.args.update(kwargs)
        for required in ('model_dir','pretrained_model'):
            assert required in kwargs
        self.labels = []
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        self.logger = self.args.logger
        self.logger.info("Initialising Tagger")

        self.set_seed()

        # Load pretrained model and tokenizer
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        #make model dir if it does not exist
        if not os.path.exists(self.args.model_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.model_dir)

        self.load_labels()

        self.args.model_type = self.args.model_type.lower()
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.config = self.config_class.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.pretrained_model,
            num_labels=self.num_labels,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.pretrained_model,
            do_lower_case=self.args.do_lower_case,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.model = self.model_class.from_pretrained(
            self.args.pretrained_model,
            from_tf=bool(".ckpt" in self.args.pretrained_model),
            config=self.config,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )

        if self.args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.to(self.args.device)

        self.logger.info("Tagger initialisation complete")

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_labels(self, labels, pad_token_label_id=None):
        self.labels = labels
        if pad_token_label_id is None:
            # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
            self.pad_token_label_id = CrossEntropyLoss().ignore_index
        else:
            self.pad_token_label_id = pad_token_label_id

    def train(self, train_file):
        """ Train the model """
        self.logger.info("Training on file %s", train_file)

        train_dataset = self.load_and_cache_examples(train_file, mode="train")

        if self.args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.args.pretrained_model, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.pretrained_model, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.args.pretrained_model, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.pretrained_model, "scheduler.pt")))

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True
            )

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.pretrained_model):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(self.args.pretrained_model.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.args.gradient_accumulation_steps)

            self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info("  Continuing training from epoch %d", epochs_trained)
            self.logger.info("  Continuing training from global step %d", global_step)
            self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0]
        )
        self.set_seed()  # Added here for reproductibility
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids

                if self.args.debug:
                    logger.info("DEBUG train input instance, tensor dimensions: %s", [(k,v.size()) for k,v in inputs.items()])
                    logger.info("DEBUG train input instance: %s", inputs)
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        # Log metrics
                        if (
                            self.args.local_rank == -1 and self.args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results, _ = self.evaluate(mode="dev")
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.logging_steps, global_step)
                        logging_loss = tr_loss

                    if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # Save model checkpoint
                        model_dir = os.path.join(self.args.model_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(model_dir)
                        self.tokenizer.save_pretrained(model_dir)

                        param_file = os.path.join(model_dir, "training_self.args.bin")
                        logger.info("Saving parameters to %s", param_file)
                        torch.save({ k:v for k,v in self.args.items() if v is None or isinstance(v, (str,float,int,bool))} , param_file)

                        logger.info("Saving model checkpoint to %s", model_dir)

                        torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(model_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", model_dir)

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.args.local_rank in [-1, 0]:
            tb_writer.close()

        self.logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        return global_step, tr_loss / global_step


    def evaluate(self, datafile, mode, prefix=""):
        eval_dataset = self.load_and_cache_examples(datafile, mode=mode)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if self.args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # multi-gpu evaluate
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        self.logger.info("***** Running evaluation %s *****", prefix)
        self.logger.info("  Num examples = %d", len(eval_dataset))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)
        self.logger.info("  Label set size = %d", self.num_labels)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                if self.args.debug:
                    logger.info("DEBUG eval input instance, tensor dimensions: %s", [(k,v.size()) for k,v in inputs.items()])
                    logger.info("DEBUG eval input instance: %s", inputs)
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if self.args.debug:
                    logger.info("DEBUG eval output logits: %s", logits)

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            if self.args.debug:
                logger.info("DEBUG eval output label ids: %s", out_label_ids)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        if self.args.debug:
            logger.info("DEBUG eval label map: %s", label_map)
            logger.info("DEBUG eval output label list: %s", out_label_list)

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "instancecount": len(preds_list),
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list


    def load_labels(self):
        labels_file = self.labels_file()
        if self.args.labels_file:
            if not os.path.exists(labels_file):
                #copy the explicitly specified label file
                logger.info("Copying labels file %s", self.args.labels_file)
                shutil.copyfile(self.args.label_file, labels_file)
        if os.path.exists(labels_file):
            self.labels = []
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    label = line.strip()
                    if label:
                        self.labels.append(label)
            self.num_labels = len(self.labels)
        else:
            logger.info("Extracting labels from training data %s", self.args.train_file)
            examples = TaggerInputDataset(logger,labelsonly=True)
            try:
                examples.load_mbt_file(self.args.train_file)
            except (AttributeError, KeyError):
                raise Exception("Model has not been trained yet, no labels found in model directory {}".format(self.args.model_dir))
            self.labels = examples.labels()
            self.num_labels = examples.num_labels()
            self.save_labels()

        self.logger.info("Loaded %d labels", self.num_labels)

    def labels_file(self):
        return os.path.join(
            self.args.model_dir,
            "{}.labels".format( list(filter(None, self.args.pretrained_model.split("/"))).pop()
            ),
        )

    def save_labels(self):
        labels_file = self.labels_file()
        logger.info("Saving %d labels to %s", self.num_labels, labels_file)
        with open(labels_file, 'w', encoding='utf-8') as f:
            for label in self.labels:
                f.write(label + "\n")

    def load_and_cache_examples(self, datafile, mode):
        if self.args.local_rank not in [-1, 0] and mode == 'train':# and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            self.args.model_dir,
            "{}_{}_{}.features.bin".format(
                mode, list(filter(None, self.args.pretrained_model.split("/"))).pop(), str(self.args.max_seq_length)
            ),
        )

        if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            logger.info("Loading labels from cached file %s", cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", datafile)
            examples = TaggerInputDataset(logger)
            examples.load_mbt_file(datafile)
            examples.convert_to_features(
                self.args.max_seq_length,
                self.tokenizer,
                cls_token_at_end=bool(self.args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=2 if self.args.model_type in ["xlnet"] else 0,
                sep_token=self.tokenizer.sep_token,
                sep_token_extra=bool(self.args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(self.args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.args.model_type in ["xlnet"] else 0,
                pad_token_label_id=self.pad_token_label_id,
            )
            if self.args.local_rank in [-1, 0]:
                logger.info("saving features into cached file %s", cached_features_file)
                torch.save(examples.features, cached_features_file)
            features = examples.features

        if self.args.local_rank == 0 and mode == 'train': # and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def __call__(self, train_file=None,dev_file=None,test_file=None):
        self.logger.info("Calling tagger for train_file=%s, dev_file=%s, test_file=%s", train_file,dev_file,test_file)
        self.logger.info(" with tagger parameters: %s", self.args)

        # Training
        if train_file:

            #train
            self.train(train_file)

            # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
            if self.args.local_rank == -1 or torch.distributed.get_rank() == 0:
                # Create output directory if needed
                if not os.path.exists(self.args.model_dir) and self.args.local_rank in [-1, 0]:
                    os.makedirs(self.args.model_dir)

                self.logger.info("Saving model checkpoint to %s", self.args.model_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(self.args.model_dir)
                self.tokenizer.save_pretrained(self.args.model_dir)

                # Good practice: save your training arguments together with the trained model
                torch.save({ k:v for k,v in self.args.items() if v is None or isinstance(v, (str,float,int,bool))} , os.path.join(self.args.model_dir, "training_self.args.bin"))

        # Evaluation on dev set
        dev_evaluation = {}
        if dev_file and self.args.local_rank in [-1, 0]:
            self.tokenizer = self.tokenizer_class.from_pretrained(self.args.model_dir, do_lower_case=self.args.do_lower_case)
            checkpoints = [self.args.model_dir]
            if self.args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(self.args.model_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getself.logger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            self.logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                self.model = self.model_class.from_pretrained(checkpoint)
                self.model.to(self.args.device)
                result, _ = self.evaluate(dev_file, mode="dev", prefix=global_step)
                if global_step:
                    result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
                dev_evaluation.update(result)
            dev_filename = os.path.basename(dev_file)
            output_eval_file = dev_filename + "_evaluation.log"
            with open(output_eval_file, "w") as writer:
                for key in sorted(dev_evaluation.keys()):
                    writer.write("{} = {}\n".format(key, str(dev_evaluation[key])))

        test_output = []
        test_evaluation = {}
        if test_file and self.args.local_rank in [-1, 0]:
            self.tokenizer = self.tokenizer_class.from_pretrained(self.args.model_dir, do_lower_case=self.args.do_lower_case)
            self.model = self.model_class.from_pretrained(self.args.model_dir)
            self.model.to(self.args.device)
            result, predictions = self.evaluate(test_file, mode="test")

            # Save evaluation results
            test_filename = os.path.basename(test_file)
            output_eval_file = test_filename + "_evaluation.log"
            with open(output_eval_file, "w") as writer:
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))
            test_evaluation = result

            if self.args.debug:
                self.logger.info("DEBUG predictions: %s", predictions)

            # Output predictions to standard output
            test_output = []
            with open(test_file, "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n" or line == "<utt>":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        test_output_item = (line.split("\t")[0], predictions[example_id].pop(0))
                        sys.stdout.write("{}\t{}\n".format(test_output_item[0], test_output_item[1]))
                    else:
                        self.logger.warning("No prediction for '%s' (maximum length exceeded?)", line.split()[0])

        return dev_evaluation, test_output, test_evaluation


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=False,
        help="The file containing the trainings data (two tab separated columns (word, token), one token per line, sentences delimited by an empty line or <utt>)",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        required=False,
        help="The file containing the development data (two tab separated columns (word, token), one token per line, sentences delimited by an empty line or <utt>). Evaluation will be performed on this file.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        required=False,
        help="The file containing the development data (two tab separated columns (word, token), one token per line, sentences delimited by an empty line or <utt>)",
    )

    #parser.add_argument(
    #    "--work_dir",
    #    default=".",
    #    type=str,
    #    required=False,
    #    help="The directory used for caching fine-tuned models",
    #)
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--pretrained_model",
        "-m",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--model_dir",
        "-d",
        default=None,
        type=str,
        required=True,
        help="The directory where the fine-tuned model will be stored (and checkpoints) will be written and read from.",
    )
    parser.add_argument(
        "--labels_file",
        "-l",
        default=None,
        type=str,
        required=False,
        help="Path to a file holding all labels (plain text, utf-8, one label per line), will be automatically derived from training data if not specified."
    )

    #Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as pretrained_model"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as pretrained_model",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained transformer models downloaded from huggingface?",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_model_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--debug","-D", help="Enables debug mode for extra verbose output", action="store_true")
    args = parser.parse_args()

    if (
        os.path.exists(args.model_dir)
        and os.listdir(args.model_dir)
        and args.train_file
        and not args.overwrite_model_dir
    ):
        raise ValueError(
            "Model output directory ({}) already exists and is not empty. Use --overwrite_model_dir to overcome.".format(
                args.model_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    kwargs = args.__dict__
    kwargs['logger'] = logger

    tagger = Tagger(**kwargs)
    tagger(args.train_file, args.dev_file, args.test_file)


if __name__ == "__main__":
    main()
