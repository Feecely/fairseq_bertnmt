# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import bert

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from bert import BertModel, BertTokenizer, BertForMaskedLM
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import BartForConditionalGeneration, BartTokenizer, BertModel, BartModel, BartConfig, BertLayer, BertConfig
from transformers import ElectraTokenizer, ElectraModel, ElectraForPreTraining, ElectraForMaskedLM

AAA = 0

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


@register_model("transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        def spm(path):
            return {
                'path': path,
                'bpe': 'sentencepiece',
                'tokenizer': 'space',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            'transformer.wmt20.en-iu.news': spm(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            'transformer.wmt20.en-iu.nh': spm(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            'transformer.wmt20.iu-en.news': spm(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            'transformer.wmt20.iu-en.nh': spm(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            'transformer.flores101.mm100.615M': spm(
                'https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            'transformer.flores101.mm100.175M': spm(
                'https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.mask_cls_sep = getattr(args, 'mask_cls_sep', False)
        self.use_bertinput = getattr(args, 'use_bertinput', False)
        self.use_bartinput = getattr(args, 'use_bartinput', False)
        self.use_electrainput = getattr(args, 'use_electrainput', False)
        self.mask_lm = getattr(args, 'mask_lm', False)
        self.extra_data = getattr(args, 'extra_data', False)
        self.text_filling = getattr(args, 'text_filling', False)
        self.electra_pretrain_task = getattr(args, 'electra_pretrain_task', False)
        self.electra_pretrain_task_generator = getattr(args, 'electra_pretrain_task_generator', False)
        self.electra_generator = getattr(args, 'electra_generator', False)
        self.bert_ner = getattr(args, 'bert_ner', False)
        self.bert_sst = getattr(args, 'bert_sst', False)
        self.origin_kd = getattr(args, 'origin_kd', False)
        self.origin_kd_bart = getattr(args, 'origin_kd_bart', False)
        self.origin_kd_electra = getattr(args, 'origin_kd_electra', False)
        self.bart_decoder = getattr(args, 'bart_decoder', False)
        self.bart_decoder_init = getattr(args, 'bart_decoder_init', False)
        self.bart_decoder_freeze = getattr(args, 'bart_decoder_freeze', False)
        self.bert_auto_encoder = getattr(args, 'bert_auto_encoder', 0)
        self.bert_auto_bertencoder = getattr(args, 'bert_auto_bertencoder', 0)
        self.bart_auto_bartdecoder = getattr(args, 'bart_auto_bartdecoder', 0)

        self.berttokenizer = BertTokenizer.from_pretrained(args.bert_model_name, do_lower_case=False)
        if self.use_bartinput:
            self.barttokenizer = BartTokenizer.from_pretrained(args.bart_model_name, do_lower_case=False)
        if self.use_electrainput:
            self.electratokenizer = ElectraTokenizer.from_pretrained(args.electra_model_name)
        if self.origin_kd is True:
            model_name = args.bert_model_name
            self.bertmasklm = BertModel.from_pretrained(model_name)
            bert_dim = self.bertmasklm.embeddings.word_embeddings.embedding_dim
            enc_dim = args.encoder_embed_dim
            if enc_dim != bert_dim:
                self.transform_fc = nn.Linear(enc_dim, bert_dim)

        if self.origin_kd_bart is True:
            model_name = args.bart_model_name
            self.bartmasklm = BartModel.from_pretrained(model_name)
            bart_dim = self.bartmasklm.config.d_model
            enc_dim = args.encoder_embed_dim
            if enc_dim != bart_dim:
                self.transform_fc = nn.Linear(enc_dim, bart_dim)

        if self.origin_kd_electra is True:
            model_name = args.electra_model_name
            self.electramasklm = ElectraModel.from_pretrained(model_name)
            electra_dim = self.electramasklm.embeddings.word_embeddings.embedding_dim
            enc_dim = args.encoder_embed_dim
            if enc_dim != electra_dim:
                self.transform_fc = nn.Linear(enc_dim, electra_dim)

        if self.mask_lm is True:
            model_name = args.bert_model_name
            # import pdb; pdb.set_trace()
            self.bertmasklm = BertForMaskedLM.from_pretrained(model_name)  # bertmasklm.cls.predictions.decoder

            # args.bert_out_dim = self.bertmasklm.bert.hidden_size

            self.mask_fc1 = self.bertmasklm.cls
            self.mask_fc1.requires_grad = False
            
            # self.mask_fc2 = nn.Linear(args.encoder_embed_dim, len(self.encoder.dictionary))
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            
            fc2_in_dim = args.encoder_embed_dim
            # choose between MT encoder layers and bert encoder layers
            assert self.bert_auto_encoder * self.bert_auto_bertencoder == 0
            if self.bert_auto_encoder > 0:
                self.bert_auto_encoder_layers = nn.ModuleList([])
                self.bert_auto_encoder_layers.extend(
                    [self.encoder.build_encoder_layer(args) for i in range(self.bert_auto_encoder)])
                 
            elif self.bert_auto_bertencoder > 0:
                self.endim2bertdim = nn.Linear(args.encoder_embed_dim, self.bertmasklm.config.hidden_size)
                configuration = BertConfig.from_json_file(model_name + '/config.json')
                # change number of layers
                configuration.num_hidden_layers = self.bert_auto_bertencoder
                bert_auto_bertencoder_model = BertModel(configuration)
                self.bert_auto_bertencoder_layers = nn.ModuleList([])
                self.bert_auto_bertencoder_layers.extend(
                    [copy.deepcopy(bert_auto_bertencoder_model.encoder.layer[i]) for i in range(self.bert_auto_bertencoder)])
                fc2_in_dim = self.bertmasklm.config.hidden_size
                del bert_auto_bertencoder_model
                
            self.mask_fc2 = nn.Linear(fc2_in_dim, len(self.berttokenizer.vocab))

        if self.text_filling is True:
            model_name = args.bart_model_name
            self.bart_tokenizer = BartTokenizer.from_pretrained(model_name, do_lower_case=False)
            self.bartmasklm = BartForConditionalGeneration.from_pretrained(model_name)

            self.bart_mask_fc2 = nn.Linear(args.encoder_embed_dim, self.bart_tokenizer.vocab_size)
            # self.bart_mask_fc2 = nn.Linear(args.encoder_embed_dim, len(self.encoder.dictionary))
            self.bart_loss_fct = nn.CrossEntropyLoss(ignore_index=1, reduction='sum')
            if self.bart_decoder:
                bart_dim = self.bartmasklm.config.d_model
                enc_dim = args.encoder_embed_dim
                self.bart_fc = nn.Linear(enc_dim, bart_dim)
                if self.bart_decoder_init and self.bart_auto_bartdecoder > 0:
                    configuration = BartConfig.from_json_file(model_name + '/config.json')
                    # assert self.bart_auto_bartdecoder > 0
                    configuration.num_hidden_layers = self.bart_auto_bartdecoder
                    # bert_auto_bertencoder_model = BertModel(configuration)
                    # self.bert_auto_bertencoder_layers = nn.ModuleList([])
                    # self.bert_auto_bertencoder_layers.extend(
                    # [copy.deepcopy(bert_auto_bertencoder_model.encoder.layer[i]) for i in range(self.bert_auto_bertencoder)])
                    tmp_model = BartForConditionalGeneration(configuration)
                    self.bart_decoder_net = copy.deepcopy(tmp_model.model.decoder)
                    self.bart_lm_head = copy.deepcopy(tmp_model.lm_head)
                    del tmp_model
                elif self.bart_decoder_init:
                    configuration = BartConfig.from_json_file(model_name + '/config.json')
                    tmp_model = BartForConditionalGeneration(configuration)
                    self.bart_decoder_net = copy.deepcopy(tmp_model.model.decoder)
                    self.bart_lm_head = copy.deepcopy(tmp_model.lm_head)
                    del tmp_model
                else:
                    self.bart_decoder_net = copy.deepcopy(self.bartmasklm.model.decoder)
                    self.bart_lm_head = copy.deepcopy(self.bartmasklm.lm_head)

                if self.bart_decoder_freeze:
                    for para in self.bart_decoder_net.parameters():
                        para.requires_grad = False
                    for para in self.bart_lm_head.parameters():
                        para.requires_grad = False

        if self.electra_pretrain_task is True:
            model_name = args.electra_model_name
            self.electra_tokenizer = ElectraTokenizer.from_pretrained(model_name)
            self.electramasklm = ElectraForPreTraining.from_pretrained(model_name)
            # self.electra_fc1 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=True)
            # self.electra_prd = nn.Linear(args.encoder_embed_dim, 1, bias=True)
            if self.electra_generator is not None:
                self.electra_generator_model = ElectraForMaskedLM.from_pretrained(self.electra_generator)
                for para in self.electra_generator_model.parameters():
                    para.requires_grad = False
            if self.electra_pretrain_task_generator:
                self.electra_ger_prd = nn.Linear(args.encoder_embed_dim, len(self.electra_tokenizer.vocab))

            else:
                self.electra_fc1 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=True)
                self.electra_prd = nn.Linear(args.encoder_embed_dim, 1, bias=True)
        if self.bert_ner is True:
            # TODO: check whether this will work.
            model_name = args.bert_model_name[:-3] + 'ner'
            self.bert_ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_fc = self.bert_ner_model.classifier
            self.ner_fc.requires_grad = False
            # import pdb; pdb.set_trace()
            self.encoder_dropout = nn.Dropout(self.bert_ner_model.config.hidden_dropout_prob)
            self.encoder_classifier = nn.Linear(self.bert_ner_model.config.hidden_size,
                                                self.bert_ner_model.config.num_labels)

        if self.bert_sst is True:
            model_name = args.bert_model_name[:-3] + 'sst'
            self.bert_sst_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sst_fc = self.bert_sst_model.classifier
            self.sst_fc.requires_grad = False
            self.encoder_dropout = nn.Dropout(self.bert_sst_model.config.hidden_dropout_prob)
            self.encoder_classifier = nn.Linear(self.bert_sst_model.config.hidden_size,
                                                self.bert_sst_model.config.num_labels)
            self.dense = nn.Linear(self.bert_sst_model.config.hidden_size, self.bert_sst_model.config.hidden_size)
            self.activation = nn.Tanh()
        # self.use_bertinput = getattr(args, 'mask_lm', False)
        # self.mask_lm = getattr(args, 'mask_lm', False)
        # self.bert_ner = getattr(args, 'bert_ner', False)
        # self.bert_sst = getattr(args, 'bert_sst', False)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )
        # parser.add_argument('--finetune-bert', action='store_true', help='...')
        parser.add_argument('--use-bertinput', action='store_true', help='...')
        parser.add_argument('--use-bartinput', action='store_true', help='...')
        parser.add_argument('--use-electrainput', action='store_true', help='...')
        parser.add_argument('--mask-lm', action='store_true', help='...')
        parser.add_argument('--bert-ner', action='store_true', help='...')
        parser.add_argument('--bert-sst', action='store_true', help='...')

        parser.add_argument('--origin-kd', action='store_true', help='...')
        parser.add_argument('--origin-kd-bart', action='store_true', help='...')
        parser.add_argument('--origin-kd-electra', action='store_true', help='...')
        parser.add_argument('--kd-alpha', default=0.9, type=float)
        parser.add_argument('--extra-data', action='store_true', help='...')
        parser.add_argument('--electra-pretrain', action='store_true', help='...')
        parser.add_argument('--electra-pretrain-task', action='store_true', help='...')
        parser.add_argument('--electra-pretrain-task-generator', action='store_true', help='...')
        parser.add_argument('--electra-generator', default=None, type=str)
        parser.add_argument('--denoising', action='store_true', help='...')
        parser.add_argument('--masking', action='store_true', help='...')
        parser.add_argument('--bert-model-name', default='bert-base-uncased', type=str)
        parser.add_argument('--bart-model-name', default='bart-base-uncased', type=str)
        parser.add_argument('--electra-model-name', default='electra-base-uncased', type=str)
        parser.add_argument('--text-filling', action='store_true', help='...')
        parser.add_argument('--bart-decoder', action='store_true', help='...')
        parser.add_argument('--bart-decoder-init', action='store_true', help='...')
        parser.add_argument('--bart-decoder-freeze', action='store_true', help='...')
        parser.add_argument('--bert-auto-encoder', default=0, type=int)
        parser.add_argument('--bert-auto-bertencoder', default=0, type=int)
        parser.add_argument('--bart-auto-bartdecoder', default=6, type=int)
        # parser.add_argument('--bart-auto-encoder', default=0, type=int)

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary


        if len(task.datasets) > 0:
            src_berttokenizer = next(iter(task.datasets.values())).berttokenizer
        else:
            src_berttokenizer = BertTokenizer.from_pretrained(args.bert_model_name, do_lower_case=False)

        if getattr(args, 'use_bartinput', False):
            if len(task.datasets) > 0:
                src_barttokenizer = next(iter(task.datasets.values())).barttokenizer
            else:
                src_barttokenizer = BartTokenizer.from_pretrained(args.bart_model_name, do_lower_case=False)

        if getattr(args, 'use_electrainput', False):
            if len(task.datasets) > 0:
                src_electratokenizer = next(iter(task.datasets.values())).electratokenizer
            else:
                src_electratokenizer = ElectraTokenizer.from_pretrained(args.electra_model_name)

        if args.share_all_embeddings:
            # if src_dict != tgt_dict:
            #     raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if getattr(args, 'use_bertinput', False):
                src_dict = src_berttokenizer
            if getattr(args, 'use_bartinput', False):
                src_dict = src_barttokenizer
            if getattr(args, 'use_electrainput', False):
                src_dict = src_electratokenizer
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        if hasattr(dictionary, 'pad_token_id'):
            padding_idx = dictionary.pad_token_id
        else:
            padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            bert_input=None,
            BERT_encoder_input=None,
            BERT_encoder_output=None,
            BART_encoder_input=None,
            BART_encoder_output=None,
            ELECTRA_encoder_input=None,
            ELECTRA_encoder_output=None,
            BERT_bert_input=None,
            BERT_bert_output=None,
            BART_bart_input=None,
            BART_bart_output=None,
            ELECTRA_electra_input=None,
            ELECTRA_electra_output=None,
            BERT_bert_labels=None,
            BERT_encoder_mapping=None,
            BART_encoder_mapping=None,
            ELECTRA_encoder_mapping=None,
            extra_data=None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if bert_input is not None:
            bert_encoder_padding_mask = bert_input.eq(self.berttokenizer.pad())
        if self.mask_cls_sep:
            bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.cls())
            bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.sep())
        if self.use_bertinput:
            bert_src_lengths = (bert_input != self.berttokenizer.pad()).sum(-1)
            encoder_out = self.encoder(bert_input, src_lengths=bert_src_lengths)
        elif self.use_bartinput:
            # import pdb; pdb.set_trace()
            bart_src_lengths = (BART_bart_output != self.barttokenizer.pad_token_id).sum(-1)
            encoder_out = self.encoder(BART_bart_output, src_lengths=bart_src_lengths)
        elif self.use_electrainput:
            # import pdb; pdb.set_trace()
            electra_src_lengths = (ELECTRA_electra_output != self.electratokenizer.pad_token_id).sum(-1)
            encoder_out = self.encoder(ELECTRA_electra_output, src_lengths=electra_src_lengths)
        else:
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
            )
        if self.mask_lm:
            if BERT_encoder_input is None or BERT_encoder_output is None or self.use_bertinput:
                BERT_encoder_input, BERT_encoder_output = BERT_bert_input, BERT_bert_output
            # MT encoder with masked input
            mask_src_lengths = (BERT_encoder_input != self.encoder.dictionary.pad_index).sum(-1)
            mask_encoder_out = self.encoder(BERT_encoder_input, mask_src_lengths)
            if self.bert_auto_encoder:
                mask_auto_encoder_out = mask_encoder_out['encoder_out'][-1]
                # run through auto encoder layers
                for layer in self.bert_auto_encoder_layers:
                    mask_auto_encoder_out = layer(mask_auto_encoder_out, encoder_padding_mask=mask_encoder_out['encoder_padding_mask'][0])
                mask_encoder_out['encoder_out'][-1] = mask_auto_encoder_out
                mask_encoder_out = mask_encoder_out['encoder_out'][-1].permute(1, 0, 2).contiguous()  # B * T * D
                # B * T * Vocab_bert
                mask_encoder_out = self.mask_fc2(mask_encoder_out)
            elif self.bert_auto_bertencoder:
                # B * T * D
                mask_auto_encoder_out = mask_encoder_out['encoder_out'][-1].permute(1, 0, 2).contiguous()
                # B * T * D_bert
                mask_auto_encoder_out = self.endim2bertdim(mask_auto_encoder_out)
                # run through auto encoder bert layers
                # float, B * T
                atten_mask = (~mask_encoder_out['encoder_padding_mask'][0]).to(mask_auto_encoder_out.dtype)
                input_shape = mask_auto_encoder_out.size()[:-1]
                # B * 1 * 1 * T
                extended_attn_mask = get_extended_attention_mask(atten_mask, input_shape, device=mask_auto_encoder_out.device, is_decoder=False)
                for layer in self.bert_auto_bertencoder_layers:
                    mask_auto_encoder_out = layer(mask_auto_encoder_out, attention_mask=extended_attn_mask)
                # D_bert -> V_bert
                mask_encoder_out = self.mask_fc2(mask_auto_encoder_out[0])
            else:
                mask_encoder_out = mask_encoder_out['encoder_out'][-1].permute(1, 0, 2).contiguous()  # B * T * D
                mask_encoder_out = self.mask_fc2(mask_encoder_out)

            BERT_encoder_label = (BERT_encoder_input != BERT_encoder_output).int()
            BERT_encoder_label = torch.mul(BERT_encoder_label, BERT_encoder_output)
            if self.mask_lm and self.text_filling:
                masked_encoder_loss = torch.tensor([0]).cuda()
            else:
                masked_encoder_loss = self.loss_fct(mask_encoder_out.view(-1, len(self.berttokenizer.vocab)),
                                                    BERT_bert_labels.view(-1))
            # masked_encoder_loss = self.loss_fct(mask_encoder_out.view(-1, len(self.encoder.dictionary)),
            #                                     BERT_encoder_label.view(-1))
            bert_mask_encoder_padding_mask = BERT_bert_output.eq(self.berttokenizer.pad())
            if self.mask_cls_sep:
                bert_mask_encoder_padding_mask += BERT_bert_output.eq(self.berttokenizer.cls())
                bert_mask_encoder_padding_mask += BERT_bert_output.eq(self.berttokenizer.sep())

            with torch.no_grad():
                _, mask_bert_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_encoder_padding_mask, masked_lm_labels=BERT_bert_labels)
                # import pdb; pdb.set_trace()
                # if self.extra_data:
                #     mask_bert_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_mask_encoder_padding_mask)
                # else:
                #     mask_bert_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_encoder_padding_mask)
                # mask_bert_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_mask_encoder_padding_mask)
                mask_bert_loss = self.loss_fct(mask_bert_out.view(-1, len(self.encoder.dictionary)),
                                                    BERT_bert_output.view(-1))
            mask_loss = mask_bert_loss  # + masked_encoder_loss

        if self.bert_ner:
            with torch.no_grad():
                ner_bert_out = self.bert_ner_model(bert_input)
            encoder_ner_out = self.encoder(bert_input, src_lengths=bert_src_lengths)
            encoder_ner_out = encoder_ner_out['encoder_out'][-1].permute(1, 0, 2).contiguous()
            encoder_ner_out = self.encoder_dropout(encoder_ner_out)
            ner_encoder_out = self.encoder_classifier(encoder_ner_out)

        if self.bert_sst:
            with torch.no_grad():
                sst_bert_out = self.bert_sst_model(bert_input)
            encoder_sst_out = self.encoder(bert_input, src_lengths=bert_src_lengths)
            encoder_sst_out = encoder_sst_out['encoder_out'][-1].permute(1, 0, 2).contiguous()
            first_token_tensor = encoder_sst_out[:, 0]
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            sst_encoder_out = self.encoder_dropout(pooled_output)
            sst_encoder_out = self.encoder_classifier(sst_encoder_out)

        if self.text_filling is True:
            #TODO:encoder_out is not none, difference between padding_mask
            bart_encoder_padding_mask = BART_bart_output.eq(self.bart_tokenizer.pad_token_id)
            if BART_encoder_input is None or BART_encoder_output is None or self.use_bartinput:
                    BART_encoder_input, BART_encoder_output = BART_bart_input, BART_bart_output
                    assert BART_encoder_input.shape == BART_encoder_output.shape
            if hasattr(self.encoder.dictionary, 'pad_index'):
                fill_src_lengths = (BART_encoder_input != self.encoder.dictionary.pad_index).sum(-1)
            else:
                fill_src_lengths = (BART_encoder_input != self.encoder.dictionary.pad_token_id).sum(-1)

            if self.bart_decoder:
                fill_encoder_out = self.encoder(BART_encoder_input, fill_src_lengths)['encoder_out'][-1]
                fill_encoder_out = self.bart_fc(fill_encoder_out)
                if self.bart_decoder_freeze:
                    with torch.no_grad():
                        fill_encoder_out = self.bart_decoder_net(input_ids=BART_encoder_output,
                                                                attention_mask=~bart_encoder_padding_mask,
                                                                encoder_hidden_states=fill_encoder_out)
                        fill_encoder_out = self.bart_lm_head(fill_encoder_out.last_hidden_state)
                else:
                    fill_encoder_out = self.bart_decoder_net(input_ids=BART_encoder_output, attention_mask=~bart_encoder_padding_mask, encoder_hidden_states=fill_encoder_out)
                    fill_encoder_out = self.bart_lm_head(fill_encoder_out.last_hidden_state)
            else:
                fill_encoder_out = self.encoder(BART_encoder_input, fill_src_lengths)
                fill_encoder_out = fill_encoder_out['encoder_out'][-1].permute(1, 0, 2).contiguous()
                fill_encoder_out = self.bart_mask_fc2(fill_encoder_out)


            # if self.mask_lm and self.text_filling:
            #     fill_encoder_loss = torch.tensor([0]).cuda()
            # else:
            #     fill_encoder_loss = self.bart_loss_fct(fill_encoder_out.view(-1, self.bart_tokenizer.vocab_size),
            #                                            BART_bart_output.view(-1))
            # fill_encoder_loss = self.bart_loss_fct(fill_encoder_out.view(-1, len(self.encoder.dictionary)),
            #                                        BART_encoder_output.view(-1))

            with torch.no_grad():
                # fill_bart_out = self.bartmasklm(BART_bart_input, attention_mask=~bart_encoder_padding_mask)['logits']
                # fill_bart_out = self.bartmasklm(input_ids=BART_bart_input, attention_mask=~bart_encoder_padding_mask, decoder_input_ids=BART_bart_output, decoder_attention_mask=~bart_encoder_padding_mask)['logits']
                fill_bart_out = self.bartmasklm(input_ids=BART_bart_input, attention_mask=~bart_encoder_padding_mask, labels=BART_bart_output)
                fill_bart_logits = fill_bart_out['logits']
                fill_loss = fill_bart_out['loss']
                # fill_loss = self.bart_loss_fct(fill_bart_out.view(-1, self.bart_tokenizer.vocab_size),
                                    #    BART_bart_output.view(-1))
            # print(self.bartmasklm.lm_head.state_dict())
            # fill_loss = fill_encoder_loss  # + fill_bart_loss

        if self.electra_pretrain_task is True:
            #TODO: now only suitable for use_electrainput is Ture
            if self.electra_generator is not None:
                ELECTRA_electra_input_bos = ELECTRA_electra_input == torch.tensor(101)
                ELECTRA_electra_input_eos = ELECTRA_electra_input == torch.tensor(102)
                ELECTRA_electra_input_pad = ELECTRA_electra_input == torch.tensor(0)
                ELECTRA_electra_input_sp_token = ELECTRA_electra_input_bos ^ ELECTRA_electra_input_eos ^ ELECTRA_electra_input_pad
                with torch.no_grad():
                    ELECTRA_electra_generator_output = self.electra_generator_model(ELECTRA_electra_input, labels=ELECTRA_electra_output)
                ELECTRA_electra_generator_output = ELECTRA_electra_generator_output['logits'].softmax(dim=2).max(dim=2).indices
                ELECTRA_electra_generator_output = ELECTRA_electra_input * ELECTRA_electra_input_sp_token + ELECTRA_electra_generator_output * (~ELECTRA_electra_input_sp_token)
                ELECTRA_electra_input = ELECTRA_electra_generator_output
                ELECTRA_encoder_input = ELECTRA_electra_generator_output
            else:
                ELECTRA_electra_input = ELECTRA_electra_output
                ELECTRA_encoder_input = ELECTRA_encoder_output
            assert ELECTRA_electra_input.shape == ELECTRA_electra_output.shape
            electra_labels = ELECTRA_electra_input != ELECTRA_electra_output
            electra_encoder_padding_mask = ELECTRA_electra_output.eq(self.electra_tokenizer.pad_token_id)
            electra_src_lengths = (ELECTRA_encoder_input != self.encoder.dictionary.pad_token_id).sum(-1)
            electra_encoder_out = self.encoder(ELECTRA_encoder_input, electra_src_lengths)
            electra_encoder_out = electra_encoder_out['encoder_out'][-1].permute(1, 0, 2).contiguous()
            if self.electra_pretrain_task_generator:
                electra_encoder_generator_out = self.electra_ger_prd(electra_encoder_out).softmax(dim=2).max(dim=2).indices
                electra_encoder_out = ELECTRA_electra_input * ELECTRA_electra_input_sp_token + electra_encoder_generator_out * (~ELECTRA_electra_input_sp_token)
                with torch.no_grad():
                    electra_encoder_out = self.electramasklm(input_ids=electra_encoder_out, attention_mask=~electra_encoder_padding_mask, labels=electra_labels)
                    electra_encoder_out_loss = electra_encoder_out['loss']
                    electra_encoder_out = electra_encoder_out['logits']
            else:
                electra_encoder_out = self.electra_fc1(electra_encoder_out)
                electra_encoder_out = self.electra_prd(electra_encoder_out).squeeze(-1)

            with torch.no_grad():
                electra_task_out = self.electramasklm(input_ids=ELECTRA_electra_input, attention_mask=~electra_encoder_padding_mask, labels=electra_labels)
            electra_task_loss = electra_task_out['loss']
            electra_task_out = electra_task_out['logits']

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        ret = {}
        if self.origin_kd:
            if getattr(self, "transform_fc", None) is not None:
                ret['distillation_out'] = self.transform_fc(encoder_out['encoder_out'][0])
            else:
                ret['distillation_out'] = encoder_out
            with torch.no_grad():
                # hidden or log-probs?
                # bert_encoder_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_encoder_padding_mask)
                bert_encoder_out = self.bertmasklm(bert_input, attention_mask=~bert_encoder_padding_mask)
            ret['bert_encoder_out'] = bert_encoder_out['last_hidden_state']
        if self.origin_kd_bart:
            if getattr(self, "transform_fc", None) is not None:
                ret['distillation_out'] = self.transform_fc(encoder_out['encoder_out'][0])
            else:
                ret['distillation_out'] = encoder_out
            with torch.no_grad():
                # hidden or log-probs?
                # bert_encoder_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_encoder_padding_mask)
                bart_encoder_padding_mask = BART_bart_output.eq(self.barttokenizer.pad_token_id)
                bert_encoder_out = self.bartmasklm(BART_bart_output, attention_mask=~bart_encoder_padding_mask)
            ret['bert_encoder_out'] = bert_encoder_out['encoder_last_hidden_state']

        if self.origin_kd_electra:
            if getattr(self, "transform_fc", None) is not None:
                ret['distillation_out'] = self.transform_fc(encoder_out['encoder_out'][0])
            else:
                ret['distillation_out'] = encoder_out
            with torch.no_grad():
                # hidden or log-probs?
                # bert_encoder_out = self.bertmasklm(BERT_bert_input, attention_mask=~bert_encoder_padding_mask)
                electra_encoder_padding_mask = ELECTRA_electra_output.eq(self.electratokenizer.pad_token_id)
                bert_encoder_out = self.electramasklm(ELECTRA_electra_output, attention_mask=~electra_encoder_padding_mask)
            ret['bert_encoder_out'] = bert_encoder_out['last_hidden_state']

        if self.mask_lm:
            ret['mask_bert_out'] = mask_bert_out
            ret['mask_encoder_out'] = mask_encoder_out
            ret['mask_loss'] = mask_loss
            ret['BERT_bert_labels'] = BERT_bert_labels
            ret['BERT_encoder_mapping'] = BERT_encoder_mapping
            ret['bert_padding_mask'] = bert_mask_encoder_padding_mask

        if self.text_filling:
            ret['fill_bart_out'] = fill_bart_logits
            ret['fill_encoder_out'] = fill_encoder_out
            ret['fill_loss'] = fill_loss
            ret['BART_encoder_mapping'] = BART_encoder_mapping
            ret['bart_padding_mask'] = bart_encoder_padding_mask

        if self.electra_pretrain_task:
            ret['electra_encoder_out'] = electra_encoder_out
            ret['electra_task_out'] = electra_task_out
            ret['electra_task_loss'] = electra_task_loss

        if self.bert_ner:
            ret['ner_bert_out'] = ner_bert_out
            ret['ner_encoder_out'] = ner_encoder_out

        if self.bert_sst:
            ret['sst_bert_out'] = sst_bert_out
            ret['sst_encoder_out'] = sst_encoder_out

        return decoder_out, ret

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens,
                                       src_lengths,
                                       return_all_hiddens,
                                       token_embeddings)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())
        # occur error
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = getattr(args, "base_layers", 0)
        for i in range(num_base_layers):
            self.layers.insert(((i + 1) * args.decoder_layers) // (num_base_layers + 1), BaseLayer(args))

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                    enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: torch.device, is_decoder: bool) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.
    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

@register_model_architecture("transformer", "transformer_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    return base_architecture(args)


@register_model_architecture("transformer", "transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("transformer", "transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_768")
def transformer_wmt_en_de(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    # args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_1024")
def transformer_wmt_en_de(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    # args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)
