# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import math
import pdb

from . import data_utils, FairseqDataset

class DenoisingBartDataset(FairseqDataset):

    def __init__(
        self, src, src_sizes, src_dict,
        srcbart=None, srcbart_sizes=None, barttokenizer=None, map_dataset=None,
        mask_whole_words=None, item_transform_func=None,
        left_pad_source=True, left_pad_target=False,
        shuffle=False, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        self.src = src
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.srcbart = srcbart
        self.map_dataset = map_dataset
        self.srcbart_sizes = np.array(srcbart_sizes) if srcbart_sizes is not None else None
        self.barttokenizer = barttokenizer
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.mask_idx = self.barttokenizer.convert_tokens_to_ids('<mask>')
        # self.mask_idx = self.src_dict.index('<mask>')
        self.encoder_mask_idx = self.src_dict.index('<mask>')
        # self.mask_idx = self.barttokenizer._convert_token_to_id('<mask>')
        self.mask_whole_word = mask_whole_words
        self.mask_ratio = 0.3
        self.random_ratio = 0.1
        self.insert_ratio = 0.0
        self.rotate_ratio = 0.5
        self.permute_sentence_ratio = 1.0
        self.bart_eos = self.barttokenizer.convert_tokens_to_ids(self.barttokenizer.eos_token)
        self.full_stop_index = self.bart_eos
        self.item_transform_func = item_transform_func

        self.mask_length = 'span-poisson'
        self.poisson_lambda = 3.5
        self.replace_length = -1

        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)
        
        src_sizes = np.reshape(self.src_sizes, [-1, 1]) if len(self.src_sizes.shape) == 1 else self.src_sizes
        srcbart_sizes = np.reshape(self.srcbart_sizes, [-1, 1]) if len(self.srcbart_sizes.shape) == 1 else self.srcbart_sizes
        
        self.sizes = (
            np.concatenate((src_sizes, srcbart_sizes), axis=-1)
        ).max(-1)
        # init pad_dict
        if hasattr(self.src, "pad_dict"):
            self.pad_dict = self.src.pad_dict
        else:
            self.pad_dict = {}
        
        self.pad_dict['BART'] = barttokenizer.pad_token_id
        
    def __getitem__(self, index):
        # encoder_datasets = self.encoder_datasets[index]
        
        source_item = self.src[index]
        if isinstance(source_item, dict):
            ret = source_item
        else:
            # in this case, we store source with original inputs. 
            ret = {
                'source': source_item,
                'id': index,
            }
        bart_item = self.srcbart[index]
        
        def build_bart_item():
            assert bart_item[-1] == self.bart_eos
            mask_bart_item = bart_item.clone()
            extra_kwargs = {}
            if self.permute_sentence_ratio > 0.0:
                mask_bart_item, perm_infos = self.permute_sentences(mask_bart_item, self.permute_sentence_ratio)
                extra_kwargs['perm_infos'] = perm_infos

            if self.mask_ratio > 0:
                mask_bart_item, mask_infos = self.add_whole_word_mask(mask_bart_item, self.mask_ratio)
                extra_kwargs['mask_infos'] = mask_infos

            if self.insert_ratio > 0:
                # TODO: for now, insert ratio is 0
                # therefore, we do not implement insertion.
                mask_bart_item = self.add_insertion_noise(mask_bart_item, self.insert_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                mask_bart_item, roll_offset = self.add_rolling_noise(mask_bart_item)
                extra_kwargs['roll_offset'] = roll_offset
            return mask_bart_item, extra_kwargs
        
        mask_bart_item, extra_kwargs = build_bart_item()
        
        # preprocess mappings if exists
        if self.map_dataset is not None:
            assert source_item.shape[0] >= bart_item.shape[0]
            bart_mapping = self.map_dataset[index]#['bart-base']
            # convert to tensor
            # adjust with bos and eos mapping.
            try:
                assert source_item.shape[0] == len(bart_mapping) + 2
            except:
                pdb.set_trace()

            bart_mapping = torch.cat([bart_mapping.new([-1]), bart_mapping, bart_mapping.new([len(bart_item)-2])])

            # bart_mapping = torch.tensor(bart_mapping, device=bart_item.device).long()
            assert bart_mapping[-1].item() >= bart_mapping[-2].item()
            # shift right
            bart_mapping += 1
            assert bart_mapping.unique().shape == bart_item.shape
            src_bart_input = self.mapping_src_to_bart_denoise_ops(ret['source'], bart_mapping, \
               **extra_kwargs)
        else:
            bart_mapping = None
            src_bart_input = source_item

        # Name format: {TASK}-{module}-{input/output}
        #TODO: check length
        #import pdb; pdb.set_trace()
        ret['BART-encoder-input'] = src_bart_input
        ret['BART-encoder-output'] = source_item
        ret['BART-bart-input'] = mask_bart_item
        ret['BART-bart-output'] = bart_item
        if bart_mapping is not None:
            ret['BART-encoder-mapping'] = bart_mapping
        return ret

    def __len__(self):
        return len(self.src)
    
    def mapping_src_to_bart_denoise_ops(self, src, src_bart_mapping, \
        perm_infos=None, mask_infos=None, insert_infos=None, roll_offset=0):
        
        result = src.clone()

        if perm_infos is not None:
            ordering = perm_infos['ordering']
            sentence_ends = perm_infos['sentence_ends']
            # Here, we only deal with one sentence cases.
            assert len(ordering) == 1
            # index, i = 1, 0
            # sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            # result[index : index + sentence.size(0)] = sentence
            # result = src.clone()
            pass

        if mask_infos is not None:
            if self.mask_span_distribution is not None:
                to_keep = mask_infos['to_keep']
                mask_bool = mask_infos['mask_bool']
                random_bool = mask_infos['random_bool']
                # boolean vectors
                mask_align = mask_bool[src_bart_mapping]
                random_align = random_bool[src_bart_mapping]
                unchange_bool = ~(mask_align + random_align)
 
                # convert indices and length to source_item scale.
                # mask_tensor = torch.zeros_like(result).fill_(self.mask_idx)
                mask_tensor = torch.zeros_like(result).fill_(self.encoder_mask_idx)
                # init random tensor
                random_tensor = torch.zeros_like(result)
                
                torch.randint(high=len(self.src_dict), size=result.shape, out=random_tensor)
                result = random_align.long() * random_tensor + \
                    mask_align.long() * mask_tensor + \
                    unchange_bool.long() * result
            else:
                raise NotImplementedError()
        if insert_infos is not None:
            # NOTE: since insert_ratio is fixed to 0.0,
            # no impl for now.
            pass
        if roll_offset != 0:
            new_offset = (src_bart_mapping < roll_offset).sum()
            result = torch.cat(
               (result[0:1], result[new_offset:-1], result[1:new_offset], result[-1:]),
               dim=0,
            )
        return result

    def permute_sentences(self, source, p=1.0):
        full_stops = source == self.full_stop_index
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1
    
        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()
    
        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]
    
        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            index += sentence.size(0)
        perm_infos = {
            'ordering': ordering,
            'sentence_ends': sentence_ends,
        }
        return result, perm_infos
    
    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start
    
    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source
    
        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
    
            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1].item() < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)
    
            # Trim to masking budget
            i = 0
            while cum_length[i].item() < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]
    
            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            # disable insertion.
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))
    
            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio
    
        # add mask_bool and random_bool
        mask_bool = torch.zeros_like(source).bool()
        random_bool = mask_bool.clone()

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.barttokenizer), size=(mask_random.sum(),)
            )
            mask_bool[indices] = True
            random_bool[indices[mask_random]] = True
            mask_bool[indices[mask_random]] = False
        
        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.barttokenizer), size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.barttokenizer), size=(mask_random.sum(),)
                    )
                    mask_bool[indices] = True
                    random_bool[indices[mask_random]] = True
                    mask_bool[indices[mask_random]] = False
    
                assert source_length - 1 not in indices
    
        source = source[to_keep]
        
        ret_dict = {
            'to_keep': to_keep,
            'mask_bool': mask_bool,
            'random_bool': random_bool,
        }
    
        # NOTE: disable insertions..
        # if num_inserts > 0:
        #     source = self.add_insertion_noise(source, num_inserts / source.size(0))
    
        return source, ret_dict
    
    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens
    
    def add_rolling_noise(self, tokens):
        # TODO: check whether eos is included.
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens, offset
    
    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens
    
        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))
    
        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)
    
        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.barttokenizer), size=(num_random,)
        )
    
        result[~noise_mask] = tokens
    
        assert (result >= 0).all()
        return result


    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), bart_pad_idx=self.barttokenizer.pad_token,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        return max(self.src_sizes[index], self.srcbart_sizes[index])

    def size(self, index):
        return self.src_sizes[index]

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
        self.srcbart.prefetch(indices)
