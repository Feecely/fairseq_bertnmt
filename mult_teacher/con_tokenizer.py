#!/usr/bin/python
# -*- coding: utf-8 -*-
from functools import partial
import os
from requests.models import DEFAULT_REDIRECT_LIMIT, default_hooks
from requests.sessions import InvalidSchema, extract_cookies_to_jar
import torch
import pdb
from transformers import AutoTokenizer
# TODO: use the same tokenizer
from bert import BertTokenizer
from transformers.models.bart import BartTokenizer
from transformers.models.electra import ElectraTokenizer
from fairseq.binarizer import safe_readline
from collections import Counter
from fairseq import tasks
import sys
import copy
import argparse
import multiprocessing as mp
#LC_ALL=en_US.UTF-8 python con_tokenizer.py

def replaced_consumer(word, idx):
    replaced = Counter()
    if idx == dict.unk_index and word != dict.unk_word:
        replaced.update([word])

def get_tokens(line, tokenizer):
    tokens = []
    ori_tokens = []
    for t in tokenizer:
        token = t.tokenize(line)
        ori_tokens.append(token)
        # TODO: restore subword tags
        token = [word.replace('##', '').replace('▁', '').replace('Ġ', '') for word in token]
        #tokens[t.name_or_path[t.name_or_path.rfind('/') + 1:]] = token
        tokens.append(token)
    return tokens, ori_tokens

def get_min_token(token1, token2):
    sent = ''.join(token1)
    nums = []
    nums.append([len(t) for t in token1])
    nums.append([len(t) for t in token2])
    min_token = []
    while nums[0] and nums[1]:
        if nums[0][0] == nums[1][0]:
            min_token.append(nums[0].pop(0))
            nums[1].pop(0)
        elif nums[0][0] < nums[1][0]:
            t = nums[0].pop(0)
            nums[1][0] -= t
            min_token.append(t)
        else:
            t = nums[1].pop(0)
            nums[0][0] -= t
            min_token.append(t)
    min_token = [0] + min_token
    m = 0
    sentence = []
    for i in range(1, len(min_token)):
        sentence.append(sent[m: m + min_token[i]])
        m += min_token[i]
    return sentence

def get_copare(sentence, token):
    ids = []
    i = 0
    while sentence:
        if token[0] != sentence[0]:
            # if sentence[0] not in token[0]:
            if not token[0].startswith(sentence[0]):
                # import pdb; pdb.set_trace()
                raise ValueError()
            token[0] = token[0].replace(sentence[0], '', 1)
            sentence.pop(0)
            ids.append(i)
        else:
            sentence.pop(0)
            token.pop(0)
            ids.append(i)
            i += 1
    return ids

def con_input(tokenizer, sentence, nums):
    encoder_input = {
        'sentence': sentence
    }
    for i in range(len(nums)):
        encoder_input[tokenizer[i].name_or_path[tokenizer[i].name_or_path.rfind('/') + 1:]] = nums[i]
    return encoder_input

def refill_subword_tags(min_tokens, space_tokens, filling="@@"):
    space_lengths = [len(item) for item in space_tokens]
    mintok_lengths = [len(item) for item in min_tokens]
    assert sum(mintok_lengths) == sum(space_lengths)
    j = 0
    new_sent = []
    word_tot_len = space_lengths[j]

    # consider punctuation
    import string
    punctuation_set = set(string.punctuation)
    for i, tok in enumerate(min_tokens):
        cur_t = len(tok)
        assert word_tot_len >= cur_t
        if tok in punctuation_set:
            new_sent.append(tok)
            # reset when encounter punctuations.
            word_tot_len -= cur_t
            if word_tot_len != 0:
                space_lengths[j] = word_tot_len
            else:
                j += 1
                if j < len(space_lengths):
                    word_tot_len = space_lengths[j]
            continue

        if filling == '@@':
            if word_tot_len < space_lengths[j]:
                new_sent.append(filling + tok)
            else:
                new_sent.append(tok)
        elif filling == 'Ġ':
            if word_tot_len < space_lengths[j]:
                new_sent.append(tok)
            else:
                new_sent.append(filling + tok)
        else:
            raise NotImplementedError()
        word_tot_len -= cur_t
        # switch to next word
        if word_tot_len == 0 and i != len(min_tokens)-1:
            j += 1
            word_tot_len = space_lengths[j]
    return new_sent

def surrogate_func(kwargs):
    return func(**kwargs)

def func(process_lines, offset, tokenizer, add_extra_outs):
    encoder_inputs = []
    sentence_splits = []
    cnt = offset
    extra_outs = {key: [] for key in range(len(tokenizer))}
    drop_list = []
    for line in process_lines:
        cnt += 1
        if cnt % 10000 == 0:
            print('Processed {} lines.'.format(cnt))
        line = line.strip()
        #line = '{} {} {}'.format('[CLS]', line, '[SEP]')
        tokens, ori_tokens = get_tokens(line, tokenizer)
        sentence = tokens[0]
        for i in range(1, len(tokens)):
            sentence = get_min_token(sentence, tokens[i])
        nums = []
        try:
            for i in range(len(tokens)):
                nums.append(get_copare(sentence=copy.deepcopy(sentence), token=copy.deepcopy(tokens[i])))
            # re-fill subword tags
            # NOTE: bart's tokenization output maybe only a space token, 
            # in which case we have to catch it by prev-token fillings
            refil_sent = refill_subword_tags(sentence, line.strip().split(), filling='Ġ')
            sentence = refil_sent
            sentence_split = " ".join(sentence)
            encoder_input = con_input(tokenizer, sentence, nums)
            encoder_inputs.append(encoder_input)
            sentence_splits.append(sentence_split)
            if add_extra_outs is True:
                ori_text = [" ".join(item) for item in ori_tokens]
                for j, text in enumerate(ori_text):
                    extra_outs[j].append(text)
        except:
            drop_list.append("{},{}".format(cnt - 1, sentence))
    print("Number of drop sentences : {}".format(len(drop_list)))
    return encoder_inputs, sentence_splits, extra_outs, drop_list

def add_line(filename, tokenizer, add_extra_outs=False, n_process=1):
    #dict = tokenizer[0]
    print('Reading file {}.'.format(filename))
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if n_process > 1:
        p = mp.Pool(processes=n_process)
        split_lines = []
        # split_tuple = []
        offsets = []
        split_size = 100000
        params = []
        # def get_params():
        for i in range(0, len(lines), split_size):
            cur_split = lines[i:i+split_size]
            # split_tuple.append([i, i+split_size])
            # offsets.append(i)
            if cur_split:
                split_lines.append(cur_split)
            params.append({
                'process_lines': cur_split,
                'offset': i,
                'add_extra_outs': add_extra_outs,
                'tokenizer': tokenizer,
            })
        pool_ret = p.imap(surrogate_func, params)
        # reord pool returns
        new_pool_ret = [None for _ in range(4)]
        # 4 returns
        for item in pool_ret:
            for j in range(len(item)):
                if isinstance(item[j], list):
                    if new_pool_ret[j] is None:
                        new_pool_ret[j] = item[j]
                    else:
                        new_pool_ret[j] += item[j]
                elif isinstance(item[j], dict):
                    if new_pool_ret[j] is None:
                        new_pool_ret[j] = item[j]
                    else:
                        for key in item[j]:
                            new_pool_ret[j][key] += item[j][key]
        return new_pool_ret
    else:
        return func(lines, 0, tokenizer, add_extra_outs)

def save_txt(filename, sentence_splits):
    with open(filename, 'w', encoding='utf8') as f:
        for sentence_split in sentence_splits:
            f.write(str(sentence_split) + '\n')
        f.close()

def save_input_dict(filename, enc_dict):
    merge_dict = dict()
    # merge keys
    for key in enc_dict[0]:
        merge_dict[key] = [item[key] for item in enc_dict]
    
    for key in merge_dict:
        output_file = "{}.{}.map".format(filename, key)
        key_mappings = merge_dict[key]
        with open(output_file, 'w', encoding='utf8') as f:
            for mapping in key_mappings:
                str_mapping = [str(item) for item in mapping]
                line = " ".join(str_mapping)
                f.write(line + '\n')

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="mp_command_master.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--file_name", type=str, required=True,
                        help="Path of input file.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path")
    parser.add_argument("--electra_tokenizer", type=str, default="",
                        help="Path of electra tokenizer")
    parser.add_argument("--bart_tokenizer", type=str, default="",
                        help="Path of bart tokenizer")
    parser.add_argument("--bert_tokenizer", type=str, default="",
                        help="Path of bert tokenizer")
    parser.add_argument("--extra_outs", action='store_true',
                        help="...")
    parser.add_argument("--drop_path", type=str, default="",
                        help="Path of drop list")
    parser.add_argument("--n_process", type=int, default=20,
                        help="Number of process")

    return parser.parse_args(args)

def main(args):
    # filename = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt/examples/copy_translation/headtest.bert.en'
    filename = args.file_name
    tokenizer = []
    # bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=False)
    # bart_tokenizer = AutoTokenizer.from_pretrained(args.bart_tokenizer, do_lower_case=False)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=False)
    bert_tokenizer.name_or_path = args.bert_tokenizer
    bart_tokenizer = BartTokenizer.from_pretrained(args.bart_tokenizer, do_lower_case=False)
    electra_tokenizer = ElectraTokenizer.from_pretrained(args.electra_tokenizer)
    #xlnet_tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased',cache_dir='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/ft_local/bart-base')
    tokenizer.append(bert_tokenizer)
    tokenizer.append(bart_tokenizer)
    tokenizer.append(electra_tokenizer)
    #tokenizer.append(xlnet_tokenizer)
    encoder_inputs, sentence_splits, extra_outs, drop_list = add_line(filename, tokenizer, add_extra_outs=args.extra_outs, n_process=args.n_process)
    # save_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt/examples/copy_translation/encoder.en'
    save_path = args.output_path
    drop_path = args.drop_path
    # dict_save_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/bert-nmt/destdir-mult/encoder_dict'
    save_txt(save_path, sentence_splits)
    save_txt(drop_path, drop_list)
    if len(extra_outs) != []:
        save_txt(save_path + '.bert', extra_outs[0])
        save_txt(save_path + '.bart', extra_outs[1])
        save_txt(save_path + '.electra', extra_outs[1])
    output_dict_save_path = args.output_path + '.data_dict'
    save_input_dict(output_dict_save_path, encoder_inputs)

if __name__ == "__main__":
    args = parse_args()
    main(args)