# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import torch
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II
from bert import BertTokenizer
from transformers import AutoTokenizer
from transformers import BartTokenizer
from transformers import ElectraTokenizer
import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    DenoisingBartDataset,
    MaskingDataset,
    ElectrapretrainDataset,
    MaskingExtraDataset,
    DenoisingBartExtraDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    bert_model_name=None,
    bart_model_name=None,
    electra_model_name=None,
    electra_pretrain=False,
    denoising=False,
    masking=False,
    extra_data=False,
    input_mapping=False,
    mask_ratio=None,
    random_ratio=None,
    insert_ratio=None,
    rotate_ratio=None,
    permute_sentence_ratio=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=False)
    if denoising:
        bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name, do_lower_case=False)
        #bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name, do_lower_case=False)
    if electra_pretrain:
        electra_tokenizer = ElectraTokenizer.from_pretrained(electra_model_name)
    srcbert_datasets = []
    extra_datasets = []
    extra_bert_datasets = []
    extra_bert_mapping_datasets = []
    extra_bart_datasets = []
    extra_bart_mapping_datasets = []
    if denoising:
        srcbart_datasets = []
    if electra_pretrain:
        srcelectra_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, src, tgt))
            bert_mapping_prefix = os.path.join(data_path, '{}.bert.map.{}-{}.'.format(split_k, src, tgt))

            if denoising:
                bartprefix = os.path.join(data_path, '{}.bart.{}-{}.'.format(split_k, src, tgt))
                bart_mapping_prefix = os.path.join(data_path, '{}.bart.map.{}-{}.'.format(split_k, src, tgt))

            if electra_pretrain:
                electraprefix = os.path.join(data_path, '{}.electra.{}-{}.'.format(split_k, src, tgt))
                electra_mapping_prefix = os.path.join(data_path, '{}.electra.map.{}-{}.'.format(split_k, src, tgt))

            if extra_data:
                extraprefix = os.path.join(data_path, '{}.extra.{}-{}.'.format(split_k, src, tgt))
                extra_bert_prefix = os.path.join(data_path, '{}.extra.bert.{}-{}.'.format(split_k, src, tgt))
                extra_bert_mapping_prefix = os.path.join(data_path, '{}.extra.bert.map.{}-{}.'.format(split_k, src, tgt))
                extra_bart_prefix = os.path.join(data_path, '{}.extra.bart.{}-{}.'.format(split_k, src, tgt))
                extra_bart_mapping_prefix = os.path.join(data_path,'{}.extra.bart.map.{}-{}.'.format(split_k, src, tgt))


        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, tgt, src))
            bert_mapping_prefix = os.path.join(data_path, '{}.bert.map.{}-{}.'.format(split_k, src, tgt))

            if denoising:
                bartprefix = os.path.join(data_path, '{}.bart.{}-{}.'.format(split_k, tgt, src))
                bart_mapping_prefix = os.path.join(data_path, '{}.bart.map.{}-{}.'.format(split_k, src, tgt))

            if electra_pretrain:
                electraprefix = os.path.join(data_path, '{}.electra.{}-{}.'.format(split_k, src, tgt))
                electra_mapping_prefix = os.path.join(data_path, '{}.electra.map.{}-{}.'.format(split_k, src, tgt))

            if extra_data:
                extraprefix = os.path.join(data_path, '{}.extra.{}-{}.'.format(split_k, src, tgt))
                extra_bert_prefix = os.path.join(data_path, '{}.extra.bert.{}-{}.'.format(split_k, src, tgt))
                extra_bert_mapping_prefix = os.path.join(data_path,
                                                         '{}.extra.bert.map.{}-{}.'.format(split_k, src, tgt))
                extra_bart_prefix = os.path.join(data_path, '{}.extra.bart.{}-{}.'.format(split_k, src, tgt))
                extra_bart_mapping_prefix = os.path.join(data_path,
                                                         '{}.extra.bart.map.{}-{}.'.format(split_k, src, tgt))

        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        # srcbert_datasets.append(indexed_dataset.make_dataset(bertprefix + src, impl=dataset_impl,
        #                                                      fix_lua_indexing=True, ))
        # if denoising:
        #     srcbart_datasets.append(indexed_dataset.make_dataset(bartprefix + src, impl=dataset_impl,
        #                                                          fix_lua_indexing=True, ))
        # if extra_data:
        #     extra_datasets.append(indexed_dataset.make_dataset(extraprefix + src, impl=dataset_impl,
        #                                                        fix_lua_indexing=True, ))
        srcbert_datasets.append(data_utils.load_indexed_dataset(bertprefix + src, dataset_impl=dataset_impl,))
        if denoising:
            srcbart_datasets.append(data_utils.load_indexed_dataset(bartprefix + src, dataset_impl=dataset_impl,
                                                                 ))
        if electra_pretrain:
            srcelectra_datasets.append(data_utils.load_indexed_dataset(electraprefix + src, dataset_impl=dataset_impl,
                                                                    ))
        if extra_data and split == 'train':
            extra_datasets.append(data_utils.load_indexed_dataset(extraprefix + src, dataset_impl=dataset_impl,
                                                                 ))
            extra_bert_datasets.append(data_utils.load_indexed_dataset(extra_bert_prefix + src, dataset_impl=dataset_impl,
                                                                 ))
            extra_bert_mapping_datasets.append(data_utils.load_indexed_dataset(extra_bert_mapping_prefix + src, dataset_impl=dataset_impl,
                                                                       ))
            extra_bart_datasets.append(data_utils.load_indexed_dataset(extra_bart_prefix + src, dataset_impl=dataset_impl,
                                                                       ))
            extra_bart_mapping_datasets.append(data_utils.load_indexed_dataset(extra_bart_mapping_prefix + src, dataset_impl=dataset_impl,
                                                                       ))
            #import pdb; pdb.set_trace()
            assert extra_datasets != [] or extra_bert_datasets != [] or extra_bert_mapping_datasets != [] or extra_bart_datasets != [] or extra_bart_mapping_datasets != []

            #extra_datasets = extra_datasets[0]
        #import pdb; pdb.set_trace()
        src_datasets[-1] = PrependTokenDataset(src_datasets[-1], token=src_dict.bos_index)
        if extra_data and split == 'train':
            extra_datasets[-1] = PrependTokenDataset(extra_datasets[-1], token=src_dict.bos_index)
        if denoising is True:
            if input_mapping is True and split == 'train':
                bart_mapping_dataset = data_utils.load_indexed_dataset(bart_mapping_prefix + src,
                                                                       dataset_impl=dataset_impl)
            else:
                bart_mapping_dataset = None

            src_datasets[-1] = DenoisingBartDataset(
                src_datasets[-1], src_datasets[-1].sizes, src_dict,
                srcbart_datasets[-1], srcbart_datasets[-1].sizes,
                bart_tokenizer,
                map_dataset=bart_mapping_dataset,
                mask_ratio=mask_ratio,
                random_ratio=random_ratio,
                insert_ratio=insert_ratio,
                rotate_ratio=rotate_ratio,
                permute_sentence_ratio=permute_sentence_ratio,
            )

        if electra_pretrain is True:
            if input_mapping is True and split == 'train':
                electra_mapping_dataset = data_utils.load_indexed_dataset(electra_mapping_prefix + src,
                                                                       dataset_impl=dataset_impl)
            else:
                electra_mapping_dataset = None

            src_datasets[-1] = ElectrapretrainDataset(
                src_datasets[-1], src_datasets[-1].sizes, src_dict,
                srcelectra_datasets[-1], srcelectra_datasets[-1].sizes,
                electra_tokenizer,
                map_dataset=electra_mapping_dataset,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                max_source_positions=max_source_positions,
                max_target_positions=max_target_positions,
            )

        if masking is True:
            if input_mapping is True and split == 'train':
                #bert_mapping_dataset = indexed_dataset.make_dataset(bert_mapping_prefix + src, impl=dataset_impl, fix_lua_indexing=True)
                bert_mapping_dataset = data_utils.load_indexed_dataset(bert_mapping_prefix + src, dataset_impl=dataset_impl)
            else:
                bert_mapping_dataset = None
            src_datasets[-1] = MaskingDataset(
                src_datasets[-1], src_datasets[-1].sizes, src_dict,
                srcbert_datasets[-1], srcbert_datasets[-1].sizes,
                bert_tokenizer,
                map_dataset=bert_mapping_dataset,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                max_source_positions=max_source_positions,
                max_target_positions=max_target_positions,
            )

        if extra_data is True and split == 'train':

            assert input_mapping is True
            src_datasets[-1] = MaskingExtraDataset(
                src_datasets[-1], src_datasets[-1].sizes, src_dict,
                extra_datasets[-1], extra_datasets[-1].sizes,
                extra_bert_datasets[-1], extra_bert_datasets[-1].sizes,
                bert_tokenizer,
                map_dataset=extra_bert_mapping_datasets[-1],
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                max_source_positions=max_source_positions,
                max_target_positions=max_target_positions,
            )

            src_datasets[-1] = DenoisingBartExtraDataset(
                src_datasets[-1], src_datasets[-1].sizes, src_dict,
                extra_datasets[-1], extra_datasets[-1].sizes,
                extra_bart_datasets[-1], extra_bart_datasets[-1].sizes,
                bart_tokenizer,
                map_dataset=extra_bart_mapping_datasets[-1],
            )

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        # srcbert_datasets = srcbert_datasets[0]
        # if denoising:
        #     srcbart_datasets = srcbart_datasets[0]

    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    src_bart_dataset = None
    src_bert_dataset = None
    src_electra_dataset = None

    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        masking,
        src_bert_dataset,
        denoising,
        src_bart_dataset,
        src_electra_dataset,
        #extra_datasets,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class TranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    extra_data: bool = field(
        default=False, metadata={"help": "..."}
    )
    denoising: bool = field(
        default=False, metadata={"help": "..."}
    )
    masking: bool = field(
        default=False, metadata={"help": "..."}
    )
    electra_pretrain: bool = field(
        default=False, metadata={"help": "..."}
    )
    input_mapping: bool = field(
        default=False, metadata={"help": "..."}
    )
    electra_model_name: str = field(
        default='electra-base', metadata={"help": "..."}
    )
    bart_model_name: str = field(
        default='bart-base', metadata={"help": "..."}
    )
    bert_model_name: str = field(
        default='bert-base-cased', metadata={"help": "..."}
    )
    finetune_bert: bool = field(
        default=False, metadata={"help": "..."}
    )
    use_bertinput: bool = field(
        default=False, metadata={"help": "..."}
    )
    use_bartinput: bool = field(
        default=False, metadata={"help": "..."}
    )
    use_electrainput: bool = field(
        default=False, metadata={"help": "..."}
    )
    # mask_ratio = None, random_ratio = None, insert_ratio = None, rotate_ratio = None, permute_sentence_ratio = None
    mask_ratio: float = field(
        default=0.3, metadata={"help": "..."}
    )
    random_ratio: float = field(
        default=0.1, metadata={"help": "..."}
    )
    insert_ratio: float = field(
        default=0.0, metadata={"help": "..."}
    )
    rotate_ratio: float = field(
        default=0.5, metadata={"help": "..."}
    )
    permute_sentence_ratio: float = field(
        default=1.0, metadata={"help": "..."}
    )

@register_task("translation_ucr", dataclass=TranslationConfig)
class TranslationTaskUcr(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.denoising = cfg.denoising if hasattr(cfg, 'denoising') else False
        self.masking = cfg.masking if hasattr(cfg, 'masking') else False
        self.electra_pretrain = cfg.electra_pretrain if hasattr(cfg, 'electra_pretrain') else False
        self.extra_data = cfg.extra_data if hasattr(cfg, 'extra_data') else False
        self.input_mapping = cfg.input_mapping if hasattr(cfg, 'input_mapping') else False
        self.bert_model_name = cfg.bert_model_name
        self.bart_model_name = cfg.bart_model_name
        self.electra_model_name = cfg.electra_model_name
        self.mask_ratio = cfg.mask_ratio,
        self.random_ratio = cfg.random_ratio,
        self.insert_ratio = cfg.insert_ratio,
        self.rotate_ratio = cfg.rotate_ratio,
        self.permute_sentence_ratio = cfg.permute_sentence_ratio,
        self.urc = []
    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        #assert split != 'train'

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            bert_model_name=self.bert_model_name,
            bart_model_name=self.bart_model_name,
            electra_model_name=self.electra_model_name,
            electra_pretrain=self.electra_pretrain,
            denoising=self.denoising,
            masking=self.masking,
            extra_data=self.extra_data,
            input_mapping=self.input_mapping,
            mask_ratio=self.mask_ratio,
            random_ratio=self.random_ratio,
            insert_ratio=self.insert_ratio,
            rotate_ratio=self.rotate_ratio,
            permute_sentence_ratio=self.permute_sentence_ratio,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample, only_task=1)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        grad_sum_trans, grad_sum_task = {}, {}
        for name, para in model.encoder.named_parameters():
            grad_sum_trans[name] = para
        model.zero_grad()

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample, only_task=2)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        for name, para in model.encoder.named_parameters():
            grad_sum_task[name] = para
        grad_sum = 0
        for name in grad_sum_trans.keys():
            grad_sum += (grad_sum_trans[name] * grad_sum_task[name]).sum()
        self.urc.append(grad_sum > 0)

        model.zero_grad()
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        # grad_sum_trans, grad_in_trans = 0, 0
        # for para in model.decoder.parameters():
        #     grad_sum_trans += para.grad.sum() if para.grad is not None else 0
        #     if para.grad is not None and para.grad.sum() > 0:
        #         grad_in_trans += para.grad.sum()
        # grad_sum_task, grad_in_task = 0, 0
        # for para in model.mask_fc1.parameters():
        #     grad_sum_task += para.grad.sum() if para.grad is not None else 0
        #     if para.grad is not None and para.grad.sum() > 0:
        #         grad_in_task += para.grad.sum()
        # for para in model.mask_fc2.parameters():
        #     grad_sum_task += para.grad.sum() if para.grad is not None else 0
        #     if para.grad is not None and para.grad.sum() > 0:
        #         grad_in_task += para.grad.sum()
        # import pdb
        # pdb.set_trace()
        # logging_output['ucr'] = grad_sum_trans * grad_sum_task > 0
        # metrics.log_scalar("ucr", 1.0 if logging_output['ucr'] else 0.0, logging_output['ntokens'], round=3)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
