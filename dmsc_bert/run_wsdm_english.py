# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import pickle
import shutil

###-------------------------global configurations-------------------------
# if True run on full training set, else run on splitted train and validation set
full_train = False

# if True run on Joey's local Machine, else run on AWS
local = True

# if english_base means bert base english, if english_large means bert large english, if chinese_base means bert base chinese
model_type = "multilanguage"

# if True use cached dataset, else generate dataset and save
cached = False

base_dir = "/home/ttx/fakenews"

load_checkpoint_flag = False
checkpoint_path = base_dir + "/models/9/chinese_epoch_0_part_train.dat"

do_train = True
do_eval = True
do_predict = False

num_epoch_train = 2

###-------------------------global configurations-------------------------


def save_checkpoint(state, filename):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(filename):
    return torch.load(filename)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class WSDM_English_Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        if full_train:
            logger.info("LOOKING AT FULL TRAINING SET {}".format(os.path.join(data_dir, "train.csv")))
            return self.create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), is_training=True)
        else:
            logger.info("LOOKING AT SPLITTED TRAINING SET {}".format(os.path.join(data_dir, "train_split.csv")))
            return self.create_examples(self._read_csv(os.path.join(data_dir, "train_split.csv")), is_training=True)

    # validation
    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "val.csv")))
        return self.create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), is_training=True)

    # test
    def get_test_examples(self, data_dir):
        return self.create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), is_training=False)

    def get_labels(self):
        return ["agreed", "disagreed", "unrelated"]

    def create_examples(self, lines, is_training=True):
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = int(line[0])
            text_a = line[5]
            text_b = line[6]
            label = None
            if is_training:
                label = line[7]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WSDM_Chinese_Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        if full_train:
            logger.info("LOOKING AT FULL TRAINING SET {}".format(os.path.join(data_dir, "train.csv")))
            return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), is_training=True)
        else:
            logger.info("LOOKING AT SPLITTED TRAINING SET {}".format(os.path.join(data_dir, "train_split.csv")))
            return self._create_examples(self._read_csv(os.path.join(data_dir, "train_split.csv")), is_training=True)

    # validation
    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "val.csv")))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), is_training=True)

    # test
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), is_training=False)

    def get_labels(self):
        return ["agreed", "disagreed", "unrelated"]

    def _create_examples(self, lines, is_training=True):
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = line[3]
            text_b = line[4]
            label = None
            if is_training:
                label = line[7]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WSDM_Multilanguage_Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        if full_train:
            logger.info("LOOKING AT FULL TRAINING SET {}".format(os.path.join(data_dir, "train.csv")))
            return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), is_training=True)
        else:
            logger.info("LOOKING AT SPLITTED TRAINING SET {}".format(os.path.join(data_dir, "train_split.csv")))
            return self._create_examples(self._read_csv(os.path.join(data_dir, "train_split.csv")), is_training=True)

    # validation
    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "val.csv")))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), is_training=True)

    # test
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), is_training=False)

    def get_labels(self):
        return ["agreed", "disagreed", "unrelated"]

    def _create_examples(self, lines, is_training=True):
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = line[3]
            text_b = line[6]
            label = None
            if is_training:
                label = line[7]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

#     def create_reverse_examples(self, lines, is_training=True):
#         examples = []
#         for (i, line) in enumerate(lines):
#             # skip header
#             if i == 0:
#                 continue
#             guid = line[0]
#             text_a = line[3]
#             text_b = line[4]
#             label = None
#             if is_training:
#                 label = line[7]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

#         for (i, line) in enumerate(lines):
#             # skip header
#             if i == 0:
#                 continue
#             guid = line[0]
#             text_a = line[4]
#             text_b = line[3]
#             label = None
#             if is_training:
#                 label = line[7]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if label id is None, means it's testing time
        label_id = None
        if example.label:
            label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.label:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def score(out, labels, weights):
    outputs = np.argmax(out, axis=1)
    label_weights = [weights[l] for l in labels]
    return np.sum((labels == outputs) * label_weights), np.sum(label_weights)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def main():
    parser = argparse.ArgumentParser()


    ## Required parameters
    parser.add_argument("--data_dir",
                        default=base_dir + "/input/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    if model_type == "english_base":
        # parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
        #                 help="Bert pre-trained model selected in the list: bert-base-uncased, "
        #                      "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        parser.add_argument("--bert_model", default=base_dir + "/english_base/", type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default="wsdm_english",
                            type=str,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default=base_dir + "/output/english_base/",
                            type=str,
                            help="The output directory where the model results will be written.")
        parser.add_argument("--model_dir",
                            default=base_dir + "/english_base/",
                            type=str,
                            help="The output directory where the model checkpoints will be written.")
    elif model_type == "english_large":
        parser.add_argument("--bert_model", default=base_dir + "/english_large/", type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default="wsdm_english",
                            type=str,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default=base_dir + "/output/english_large/",
                            type=str,
                            help="The output directory where the model results will be written.")
        parser.add_argument("--model_dir",
                            default=base_dir + "/english_large/",
                            type=str,
                            help="The output directory where the model checkpoints will be written.")
    elif model_type == "chinese":
        parser.add_argument("--bert_model", default=base_dir + "/chinese", type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default="wsdm_chinese",
                            type=str,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default=base_dir + "/output/chinese/",
                            type=str,
                            help="The output directory where the model results will be written.")
        parser.add_argument("--model_dir",
                            default=base_dir + "/chinese/",
                            type=str,
                            help="The output directory where the model checkpoints will be written.")
    elif model_type == "multilanguage":
        parser.add_argument("--bert_model", default=base_dir + "/multilanguage", type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default="wsdm_multilanguage",
                            type=str,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default=base_dir + "/output/multilanguage/",
                            type=str,
                            help="The output directory where the model results will be written.")
        parser.add_argument("--model_dir",
                            default=base_dir + "/multilanguage/",
                            type=str,
                            help="The output directory where the model checkpoints will be written.")


    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=do_train,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=do_eval,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=do_predict,
                        help="Whether to run predict on the test set.")
    parser.add_argument('--load_checkpoint_flag',
                        default=load_checkpoint_flag,
                        help='Whether to load from exisiting checkpoint')
    parser.add_argument('--checkpoint_path',
                        type=str, default=checkpoint_path,
                        help='The path of checkpoint to load from')

    if local:
        # if run locally use smaller batch sizes
        parser.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=64,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--predict_batch_size",
                            default=64,
                            type=int,
                            help="Total batch size for eval.")
    else:
        # on AWS 4 GPU use larger batch sizes
        parser.add_argument("--train_batch_size",
                            default=48,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=198,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--predict_batch_size",
                            default=198,
                            type=int,
                            help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=num_epoch_train,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=60,
                        help="random seed for initialization")

    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')


    if full_train:
        parser.add_argument('--cached_train_examples', type=str,
                            default=base_dir + '/input/' + model_type + '/cached_train_full_examples.dat')
        parser.add_argument('--cached_train_features', type=str,
                            default=base_dir + '/input/' + model_type + '/cached_train_full_features.dat')
    else:
        parser.add_argument('--cached_train_examples', type=str,
                            default=base_dir + '/input/' + model_type + '/cached_train_examples.dat')
        parser.add_argument('--cached_train_features', type=str,
                            default=base_dir + '/input/' + model_type + '/cached_train_features.dat')

    parser.add_argument('--cached_dev_examples', type=str,
                        default=base_dir + '/input/' + model_type + '/cached_dev_examples.dat')
    parser.add_argument('--cached_dev_features', type=str,
                        default=base_dir + '/input/' + model_type + '/cached_dev_features.dat')
    parser.add_argument('--cached_test_examples', type=str,
                        default=base_dir + '/input/' + model_type + '/cached_test_examples.dat')
    parser.add_argument('--cached_test_features', type=str,
                        default=base_dir + '/input/' + model_type + '/cached_test_features.dat')

    args = parser.parse_args()

    processors = {
        "wsdm_english": WSDM_English_Processor,
        "wsdm_chinese": WSDM_Chinese_Processor,
        "wsdm_multilanguage":WSDM_Multilanguage_Processor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    weights_map = {0: 1 / 15, 1: 1 / 5, 2: 1 / 16}

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None

    if args.do_train:

        if cached:
            # load
            with open(args.cached_train_examples, "rb") as reader:
                train_examples = pickle.load(reader)
        else:
            # save
            train_examples = processor.get_train_examples(args.data_dir)
            with open(args.cached_train_examples, 'wb') as writer:
                pickle.dump(train_examples, writer)

        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,
    #             cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), num_labels=len(label_list))
    #
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=len(label_list))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0

    if args.load_checkpoint_flag:
        ckpt = load_checkpoint(args.checkpoint_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer'])
        logger.info(
            "***** Resuming model and optimizer states from Checkpoint " + args.checkpoint_path + " Restart from Epoch " + str(
                start_epoch) + " *****")

    weights_tensor = torch.tensor([16/15, 16/5, 1]).to(device)

    if args.do_train:
        if cached:
            # load
            with open(args.cached_train_features, "rb") as reader:
                train_features = pickle.load(reader)
        else:
            # save
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
            with open(args.cached_train_features, 'wb') as writer:
                pickle.dump(train_features, writer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        tr_losses = []
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, weights_tensor)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

                if (step+1)%1000 == 0:
                    print("loss:", tr_loss/nb_tr_steps)

            if full_train:
                filename = args.model_dir + model_type + "_epoch_" + str(epoch) + "_full_train.dat"
            else:
                filename = args.model_dir + model_type + "_epoch_" + str(epoch) + "_part_train.dat"

            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             },
                            filename=filename)

            tr_losses.append(tr_loss / nb_tr_steps)
            print("epoch", epoch, ", tr loss:", tr_loss/nb_tr_steps)

        output_prediction_file = os.path.join(args.output_dir, "train_loss.csv")
        with open(output_prediction_file, "w") as writer:
            for i, lo in enumerate(tr_losses):
                writer.write("epoch %s, %s" % (str(i), str(lo)))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # load model temporary
        # model.load_state_dict(torch.load("/home/ttx/ttx/fakenews/models/0.8692/chinese_weight"))

        if cached:
            # load examples
            with open(args.cached_dev_examples, "rb") as reader:
                eval_examples = pickle.load(reader)

            # load features
            with open(args.cached_dev_features, "rb") as reader:
                eval_features = pickle.load(reader)
        else:

            # save examples
            eval_examples = processor.get_dev_examples(args.data_dir)
            with open(args.cached_dev_examples, 'wb') as writer:
                pickle.dump(eval_examples, writer)

            # save features
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
            with open(args.cached_dev_features, 'wb') as writer:
                pickle.dump(eval_features, writer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        # accum_weights is used for eval_score
        eval_loss, eval_accuracy, eval_score, accum_weights = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        label_inv_map = {v: k for k, v in label_map.items()}


        predictions = {}
        probs = {}
        all_guids = [f.guid for f in eval_examples]
        guid_count = 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, weights_tensor)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            tmp_eval_score, _ = score(logits, label_ids, weights_map)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            eval_score += tmp_eval_score
            # accum_weights += tmp_accum_weights

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            outputs = np.argmax(logits, axis=1)
            for i in range(len(outputs)):
                predictions[all_guids[guid_count]] = outputs[i]
                probs[all_guids[guid_count]] = logits[i]
                guid_count += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_score = eval_score / nb_eval_examples

        if args.do_train:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'eval_score':eval_score,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'eval_score': eval_score,
                      'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_prediction_file = os.path.join(args.output_dir, "eval_prediction.csv")
        with open(output_prediction_file, "w") as writer:
            logger.info("***** writing predictions to " + output_prediction_file + " *****")
            writer.write("Id,Category,Prob0,Prob1,Prob2\n")
            for key in sorted(predictions.keys()):
                writer.write("%s,%s,%f,%f, %f\n" % (str(key), str(label_inv_map[predictions[key]]), probs[key][0], probs[key][1], probs[key][2]))


    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # model.load_state_dict(torch.load("/home/ttx/ttx/fakenews/models/0.8692/chinese_weight"))

        predictions = {}

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        label_inv_map = {v: k for k, v in label_map.items()}

        if cached:
            # load examples
            with open(args.cached_test_examples, "rb") as reader:
                test_examples = pickle.load(reader)

            # load features
            with open(args.cached_test_features, "rb") as reader:
                test_features = pickle.load(reader)

        else:
            # save examples
            test_examples = processor.get_test_examples(args.data_dir)
            with open(args.cached_test_examples, 'wb') as writer:
                pickle.dump(test_examples, writer)

            # save features
            test_features = convert_examples_to_features(
                test_examples, label_list, args.max_seq_length, tokenizer)
            with open(args.cached_test_features, 'wb') as writer:
                pickle.dump(test_features, writer)

        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.predict_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)

        all_guids = [f.guid for f in test_examples]
        guid_count = 0

        model.eval()

        nb_test_steps, nb_test_examples = 0, 0

        for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)

            for output in outputs:
                predictions[all_guids[guid_count]] = output
                guid_count += 1

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        print("Total number of test examples: " + str(nb_test_examples) + " Total number of test steps: " + str(
            nb_test_steps))

        output_prediction_file = os.path.join(args.output_dir, "prediction.csv")
        with open(output_prediction_file, "w") as writer:
            logger.info("***** writing predictions to " + output_prediction_file + " *****")
            writer.write("Id,Category\n")
            for key in sorted(predictions.keys()):
                writer.write("%s,%s\n" % (str(key), str(label_inv_map[predictions[key]])))


if __name__ == "__main__":
    main()
