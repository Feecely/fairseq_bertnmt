from transformers import AlbertTokenizer, AlbertForMaskedLM, AutoTokenizer, AutoModel, BertForMaskedLM, BertModel, BertTokenizer
import torch
from fairseq.binarizer import safe_readline
def remove_lines(filename, save_path, drop_list):
    with open(filename, 'r', encoding='utf-8') as f:
        line = safe_readline(f)
        reduce_line = []
        num = 0
        j = 0
        print(len(drop_list))
        while line:
            if j >= len(drop_list) or num == int(drop_list[j]):
                j += 1
            else:
                reduce_line.append(line)
            num += 1
            if num % 100000 == 0:
                print(num)
            line = f.readline()
    filename_reduce = save_path + filename.split('/')[-1]
    with open(filename_reduce, 'w', encoding='utf8') as f:
        for line in reduce_line:
            f.write(str(line))
        f.close()

def reduce_words(filename, save_path, drop_filename):
    with open(drop_filename, 'r', encoding='utf-8') as f:
        line = safe_readline(f)
        drop_list = []
        while line:
            drop_list.append(line)
            line = f.readline()
    name_list = ['.en', '.de', '.bert.en', '.bert.de', '.bart.en', '.bart.de']
    for name in name_list:
        remove_lines(filename + name, save_path, drop_list)
rootpath = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/'
filename = rootpath + 'raw_data/wmt14_en_de-head1w/train'
save_path = rootpath + 'raw_data/filter_data/'
drop_filename = rootpath + 'raw_data/wmt14_en_de-head1w/train.train.dropout.bk'
reduce_words(filename, save_path, drop_filename)

# def get_min_token(token1, token2):
#     sent = ''.join(token1)
#     nums = []
#     nums.append([len(t) for t in token1])
#     nums.append([len(t) for t in token2])
#     min_token = []
#     while nums[0] and nums[1]:
#         if nums[0][0] == nums[1][0]:
#             min_token.append(nums[0].pop(0))
#             nums[1].pop(0)
#         elif nums[0][0] < nums[1][0]:
#             t = nums[0].pop(0)
#             nums[1][0] -= t
#             min_token.append(t)
#         else:
#             t = nums[1].pop(0)
#             nums[0][0] -= t
#             min_token.append(t)
#     min_token = [0] + min_token
#     m = 0
#     sentence = []
#     for i in range(1, len(min_token)):
#         sentence.append(sent[m: m + min_token[i]])
#         m += min_token[i]
#     return sentence
#
# def get_copare(sentence, token):
#     ids = []
#     i = 0
#     while sentence:
#         if token[0] != sentence[0]:
#             # if sentence[0] not in token[0]:
#             if not token[0].startswith(sentence[0]):
#                 raise ValueError()
#             token[0] = token[0].replace(sentence[0], '', 1)
#             sentence.pop(0)
#             ids.append(i)
#         else:
#             sentence.pop(0)
#             token.pop(0)
#             ids.append(i)
#             i += 1
#     return ids
#
# token1 = 'i li ke eat app le'
# token2 = 'i lk e e at a pp le'
# token1 = token1.split(' ')
# token2 = token2.split(' ')
# print(token1)
# print(token2)
# min_tokens = get_min_token(token1, token2)
# print(min_tokens)
# print(get_copare(min_tokens, token2))
