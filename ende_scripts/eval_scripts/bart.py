# from transformers import BartForConditionalGeneration, BartTokenizer
# model_name = '/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert/pretrain_models/bart-base'
# model = BartForConditionalGeneration.from_pretrained(model_name)
# tok = BartTokenizer.from_pretrained(model_name)
# # example_english_phrase = "UN Chief Says There Is No <mask> in Syria."
# example_english_phrase = "My friends <mask> they eat too many carbs."
# batch = tok(example_english_phrase, return_tensors='pt')
# generated_ids = model.generate(batch['input_ids'])
# print(tok.batch_decode(generated_ids, skip_special_tokens=True))

from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, BartConfig
import random
import torch
import numpy as np
SEED = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model_name = '/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert/pretrain_models/bart-base'
tok = BartTokenizer.from_pretrained(model_name)
#model = BartForConditionalGeneration.from_pretrained(model_name)
#import pdb; pdb.set_trace()
# print(model.lm_head.state_dict())

# TXT = "UN Chief Says There Is No <mask> to Stop Chemical Weapons in Syria."
# TXT_nts = "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria."
# TXT = "My friends are <mask> but they eat too many carbs."
# TXT = "My friends are <mask> they eat too many carbs."
# TXT_nts = "My friends are good but they eat too many carbs."
input_batch = ["<s>It <mask> retriever. My <mask> cute </s>",]
decoder_input_batch = ["</s><s>My dog is cute. It is a golden retriever",] #</s>
labels_batch = ["<s>My dog is cute. It is a golden retriever</s>",]
model = BartForConditionalGeneration.from_pretrained(model_name, force_bos_token_to_be_generated=True)
# model = BartForConditionalGeneration(BartConfig())

input_ids = tok.batch_encode_plus(input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
decoder_input_ids = tok.batch_encode_plus(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
labels = tok.batch_encode_plus(labels_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
# input_tok = tokenizer(input_batch, return_tensors='pt')
# input_ids = input_tok['input_ids']
# input_ids_nts = tokenizer([TXT_nts], return_tensors='pt')['input_ids']#[:,1:]
# import pdb; pdb.set_trace()
# logits = model(input_ids=input_ids, use_cache=False).logits
# logits = model(input_ids=input_ids, decoder_input_ids=input_ids_nts, use_cache=False).logits
# # masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
# # probs = logits[0, masked_index].softmax(dim=0)
# probs = logits[0].softmax(dim=1)
# values, predictions = probs.topk(1)
# # print(predictions)
# sent = []
# l = []
# for i, word in enumerate(predictions):
#     # print(values[i])
#     # print(tokenizer.decode(word).split())
#     sent.append(tokenizer.decode(word).split())
#     l.append(values[i])
# print(sent)
# print(l)
#print(tokenizer.decode(predictions).split())
# loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
output = model(input_ids=input_ids, labels=labels, return_dict=True)

# output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels, return_dict=True)
# import pdb; pdb.set_trace()
print(output.loss)
sent = []
l = []
# logits = output.logits[0]
# logsm = torch.nn.LogSoftmax(dim=-1)
# log_softmax = logsm(logits)
# scores, preds = logits.topk(3, dim=-1, largest=True, sorted=True)
# import pdb; pdb.set_trace()
# for i, pred in enumerate(preds):
#     # label = labels[0][i]
#     # print(values[i])
#     # print(tokenizer.decode(word).split())
#     sent.append([tok.decode(word).split() for word in pred])
#     # l.append(log_softmax[i, word].item())
#     # import pdb; pdb.set_trace()
# print(sent)
# print(l)

# from transformers import BartForConditionalGeneration, BartTokenizer
# model = BartForConditionalGeneration.from_pretrained('/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert/pretrain_models/bart-larger', force_bos_token_to_be_generated=True)
# tok = BartTokenizer.from_pretrained('/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert/pretrain_models/bart-larger')
# example_english_phrase = "UN Chief Says There Is No <mask> <mask> <mask> <mask> <mask> in Syria"
# # example_english_phrase = "UN Chief Says There to Stop Chemical Is No Plan Weapons in Syria"
# print('source:' + example_english_phrase)
# batch = tok(example_english_phrase, return_tensors='pt')
# generated_ids = model.generate(batch['input_ids'])
# print('target:' + str(tok.batch_decode(generated_ids, skip_special_tokens=True)))
# assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria']