# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

import torch
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
from typing import List, Dict
#from bert import BertTokenizer
from transformers.models.bart import BartTokenizer
from transformers.models.bert import BertTokenizer
from transformers.models.electra import ElectraTokenizer

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:
    @staticmethod
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
        avoid_tokenize=False,
    ) -> Dict[str, int]:
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        def replaced_consumer_from_pretrained(word, idx):
            if idx == dict.convert_tokens_to_ids(dict.unk_token) and word != dict.unk_token:
                replaced.update([word])

        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                # f.tell() does not always give the byte position in the file
                # sometimes it skips to a very large number
                # it is unlikely that through a normal read we go from
                # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
                # that the procedure breaks by the undeterministic behavior of
                # f.tell()
                if end > 0 and f.tell() > end and f.tell() < end + 2 ** 32:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                elif isinstance(dict, BertTokenizer) and not isinstance(dict, ElectraTokenizer):
                    line = line.strip()
                    line = '{} {} {}'.format('[CLS]', line, '[SEP]')
                    if avoid_tokenize is False:
                        tokenizedline = dict.tokenize(line)
                    else:
                        tokenizedline = line.strip().split()
                    # max-len:1000000000000
                    # print('----------bert_max-len:' + str(dict.max_len) + '----------')
                    # if len(tokenizedline) > dict.max_len:
                    #     tokenizedline = tokenizedline[:dict.max_len - 1]
                    #     tokenizedline.append('[SEP]')
                    words = dict.convert_tokens_to_ids(tokenizedline)
                    nwords = len(words)
                    ids = torch.IntTensor(nwords)
                    for i, word in enumerate(words):
                        ids[i] = word
                        replaced_consumer_from_pretrained(tokenizedline[i], word)
                elif isinstance(dict, BartTokenizer):
                    line = line.strip()
                    if avoid_tokenize is False:
                        # extra space at the end will cause weird outputs.
                        line = '{} {}{}'.format('<s>', line, '</s>')
                        tokenizedline = dict.tokenize(line)
                    else:
                        line = '{} {} {}'.format('<s>', line, '</s>')
                        tokenizedline = line.strip().split()
                    # tokenizedline = dict.tokenize(line)
                    words = dict.convert_tokens_to_ids(tokenizedline)
                    assert len(tokenizedline) == len(words)
                    nwords = len(words)
                    ids = torch.IntTensor(nwords)
                    for i, word in enumerate(words):
                        ids[i] = word
                        replaced_consumer_from_pretrained(tokenizedline[i], word)
                elif isinstance(dict, ElectraTokenizer):
                    line = line.strip()
                    line = '{} {} {}'.format('[CLS]', line, '[SEP]')
                    if avoid_tokenize is False:
                        tokenizedline = dict.tokenize(line)
                    else:
                        tokenizedline = line.strip().split()
                    # max-len:1000000000000
                    # print('----------bert_max-len:' + str(dict.max_len) + '----------')
                    # if len(tokenizedline) > dict.max_len:
                    #     tokenizedline = tokenizedline[:dict.max_len - 1]
                    #     tokenizedline.append('[SEP]')
                    words = dict.convert_tokens_to_ids(tokenizedline)
                    #
                    # import pdb; pdb.set_trace()
                    nwords = len(words)
                    ids = torch.IntTensor(nwords)
                    for i, word in enumerate(words):
                        ids[i] = word
                        replaced_consumer_from_pretrained(tokenizedline[i], word)
                elif dict is None:
                    line = line.strip()
                    words = line.split()
                    words = [int(item) for item in words]
                    nwords = len(words)
                    ids = torch.IntTensor(nwords)
                    for i, word in enumerate(words):
                        ids[i] = word
                else:
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(
        filename, alignment_parser, consumer, offset=0, end=-1
    ) -> Dict[str, int]:
        nseq = 0

        with open(PathManager.get_local_path(filename), "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks) -> List[int]:
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
