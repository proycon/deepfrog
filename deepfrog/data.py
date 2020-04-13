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
import logging

class TaggerInputDataset:
    def __init__(self, logger=None, maxtokens=50, labelsonly=False):
        self.tags = set() #list of all tags
        self.index2tag = []
        self.tag2index = {}
        self.maxtokens = maxtokens #maximum number of tokens per sentence
        self.maxlength = 0 #maximum number of subtokens per sentence (computed later)
        self.instances = [] #corresponds to sentences
        self.features = [] #instances after conversion to features
        self.labelset = set()
        self.labelsonly = labelsonly #if set, only gathers labels (updating self.labelset, do not load the actual instances)
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def load_file(self,filename):
        """Load a tab-separated training file, one token + label per line, returns tagged instances"""
        words = []
        labels = []
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                fields = line.split("\t")
                if not line or line == "<utt>": #support old mbt-style too
                    #end of sentence marker-found
                    if len(words) <= self.maxtokens:
                        if not self.labelsonly:
                            self.instances.append(TaggerInputInstance(len(self.instances), words, labels))
                    else:
                        print("Skipping sentence " + str(i+1) + " because it exceeds the maximum token limit",file=sys.stderr)
                    words = []
                    labels = []
                else:
                    try:
                        word, label = fields
                    except ValueError:
                        raise Exception("Unable to parse line " + str(i+1) + ": " + repr(fields))
                    if not self.labelsonly:
                        words.append(word)
                        labels.append(label)
                    self.labelset.add(label)
        if not self.labelsonly and words and len(words) <= self.maxtokens: #in case the final <utt> is omitted
            self.instances.append(TaggerInputInstance(len(self.instances), words, labels))


    def labels(self):
        """Returns all labels in the vocabulary, with the padding label on index 0"""
        return ["[PAD]"] + list(sorted(self.labelset))

    def num_labels(self):
        return len(self.labelset) + 1 #+1 for the padding label, which is always at index 0

    def convert_to_features(self,
                            max_seq_length,
                            tokenizer,
                            labels=None,
                            cls_token_at_end=False,
                            cls_token="[CLS]",
                            cls_token_segment_id=1,
                            sep_token="[SEP]",
                            sep_token_extra=False,
                            pad_on_left=False,
                            pad_token=0,
                            pad_token_segment_id=0,
                            pad_token_label_id=-100,
                            sequence_a_segment_id=0,
                            mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `labels` is a list containing the vocabulary.
            `cls_token_at_end` defines the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        if labels is None:
            labels = self.labels()
        label_map = {label: i for i, label in enumerate(labels) }

        for (ex_index, example) in enumerate(self.instances):
            if ex_index % 10000 == 0:
                self.logger.info("Conversion to features, example %d of %d", ex_index, len(self.instances))

            tokens = []
            label_ids = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                try:
                    label_id = label_map[label]
                except KeyError:
                    self.logger.info("Warning: Out-of-vocabulary label found: %s", label)
                    label_id = pad_token_label_id
                label_ids.extend([label_id] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            try:
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(label_ids) == max_seq_length
            except AssertionError:
                self.logger.error("Failure featurising sample %d: %d=%d=%d=%d=%d", example.id, max_seq_length, len(input_ids),len(input_mask), len(segment_ids), len(label_ids))
                self.logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                self.logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                self.logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                self.logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                self.logger.warning("skipping...")
                continue

            if ex_index < 5:
                self.logger.info("*** Example ***")
                self.logger.info("id: %s", example.id)
                self.logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                self.logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                self.logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                self.logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            self.features.append(
                TaggerInputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
            )

    def __iter__(self):
        return iter(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self,index):
        return self.instances[index]

    def write(self, fp):
        for instance in self:
            instance.write(fp)


class TaggerInputInstance:
    """A single training/test example for token classification."""

    def __init__(self, sample_id, words, labels):
        """Constructs a TaggerInputInstance.

        Args:
            sample_id: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = sample_id
        self.words = words
        self.labels = labels
        if self.labels:
            assert len(self.words) == len(self.labels)

    def append(self, word, label):
        self.words.append(word)
        if label:
            self.labels.append(label)

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        if self.labels:
            for word, label in zip(self.words, self.labels):
                yield word, label
        else:
            for word in self.words:
                yield word, "?"

    def write(self, fp):
        for word, label in self:
            fp.write(word + "\t" + label + "\n")
        fp.write("\n")

class TaggerInputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

