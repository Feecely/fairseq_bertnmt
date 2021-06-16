# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import pdb

from . import data_utils, FairseqDataset


class MaskingDataset(FairseqDataset):

    def __init__(
        self, src, src_sizes, src_dict,
        srcbert=None, srcbert_sizes=None, berttokenizer=None,map_dataset=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=False, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        self.src = src
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.srcbert = srcbert
        self.map_dataset = map_dataset
        #[len(i['sentence']) for i in self.encoder_dict]
        self.srcbert_sizes = np.array(srcbert_sizes) if srcbert_sizes is not None else None
        self.berttokenizer = berttokenizer
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        # init pad_dict
        if hasattr(self.src, "pad_dict"):
            self.pad_dict = self.src.pad_dict
        else:
            self.pad_dict = {}
        self.pad_dict['BERT'] = berttokenizer.pad()
        # self.sizes = self.src.sizes
        src_sizes = np.reshape(self.src_sizes, [-1, 1]) if len(self.src_sizes.shape) == 1 else self.src_sizes
        srcbert_sizes = np.reshape(self.srcbert_sizes, [-1, 1]) if len(self.srcbert_sizes.shape) == 1 else self.srcbert_sizes
        self.sizes = (
            np.concatenate((src_sizes, srcbert_sizes), axis=-1)
        ).max(-1)

    def __getitem__(self, index):
        src_item = self.src[index]
        if isinstance(src_item, dict):
            ret = src_item
            # retrieve source
            src_item = src_item['source']
        else:
            # in this case, we store source with original inputs. 
            ret = {
                'source': src_item,
                'id': index,
            }
        ret['BERT-bert-output'] = self.srcbert[index].clone()

        src_bert_item = self.srcbert[index]
        bert_mask_item, bert_mask_labels = self.build_mask_input(src_bert_item, self.berttokenizer, mlm_probability=0.15)
        if self.map_dataset is not None:
            bert_mapping = self.map_dataset[index]
            src_mask_item, bert_mapping = self.build_encoder_mask_input(src_item, bert_mapping, ret['BERT-bert-output'], bert_mask_item)
        else:
            # use bert input.
            src_mask_item = None
            bert_mapping = None

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                # TODO: check whether other inputs also need remove eos.
                src_item = self.src[index][:-1]

        # Name format: {TASK}-{module}-{input/output}
        # TODO: convert BERT encoder input
        if src_mask_item is not None:
            ret['BERT-encoder-output'] = src_item
            ret['BERT-encoder-input'] = src_mask_item
        else:
            ret['BERT-encoder-input'] = bert_mask_item
            # ret['BERT-encoder-output'] = src_bert_item
            ret['BERT-encoder-output'] = ret['BERT-bert-output']
        assert ret['BERT-encoder-input'].shape == ret['BERT-encoder-output'].shape

        ret['BERT-bert-input'] = bert_mask_item
        # ret['BERT-bert-output'] = src_bert_item
        ret['BERT-bert-labels'] = bert_mask_labels
        if bert_mapping is not None:
            ret['BERT-encoder-mapping'] = bert_mapping

        return ret
        
    def build_mask_input(self, src_bert_item, tokenizer, mlm_probability):
        bert_input, labels = src_bert_item, src_bert_item.clone()
        tokens_mask = tokenizer.vocab["[MASK]"]
        probability_matrix = np.full(labels.shape, mlm_probability)
        probability_matrix = probability_matrix * np.int64(bert_input > 102)
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        bert_input[indices_replaced] = tokens_mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced
        random_words = np.random.randint(len(tokenizer.vocab), size=labels.shape, dtype="i4")
        random_words = torch.LongTensor(random_words)
        bert_input[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return bert_input, labels

    def build_encoder_mask_input(self, src_item, src_bert_mapping, src_bert_item, bert_mask_item):
        mask_idx = self.src_dict.index("<mask>")
        # mask is True
        mask_bool = bert_mask_item == mask_idx
        # random is True
        random_bool = (src_bert_item != bert_mask_item) & ~mask_bool

        # get input alignment, 
        # add align eos
        bert_mapping = src_bert_mapping #['bert-base-cased-new']
        # bert_mapping = [-1, ] + bert_mapping + [len(src_bert_item) - 2]
        bert_mapping = torch.cat([bert_mapping.new([-1]), bert_mapping, bert_mapping.new([len(src_bert_item)-2])])
        
        assert bert_mapping[-1].item() >= bert_mapping[-2].item()
        # align_idx = src_item.new(bert_mapping)
        # because of sos in bert input, shift right
        bert_mapping = bert_mapping + 1
        assert bert_mapping.shape == src_item.shape
        assert bert_mapping.unique().shape == src_bert_item.shape

        # boolean vectors
        mask_align = mask_bool[bert_mapping]
        random_align = random_bool[bert_mapping]
        unchange_bool = ~(mask_align + random_align)
        
        mask_tensor = torch.zeros_like(src_item).fill_(mask_idx)
        # init random tensor
        random_tensor = torch.zeros_like(src_item)

        torch.randint(high=len(self.src_dict), size=src_item.shape, out=random_tensor)
        src_mask_item = random_align.long() * random_tensor + \
            mask_align.long() * mask_tensor + \
            unchange_bool.long() * src_item
        # TODO: mask labels
        return src_mask_item, bert_mapping

    def __len__(self):
        return len(self.src)

    def num_tokens(self, index):
        return max(self.src_sizes[index], self.srcbert_sizes[index])

    def size(self, index):
        # add --max-positions to filter too long sentences.
        return max(self.src_sizes[index], self.srcbert_sizes[index])

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.srcbert.prefetch(indices)
