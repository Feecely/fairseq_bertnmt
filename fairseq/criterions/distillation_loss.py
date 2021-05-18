# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
from bert import BertModel
from fairseq import utils
import torch.nn as nn
import torch
from . import FairseqCriterion, register_criterion


@register_criterion('distillation_loss')
class DistillationLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        self.MSE_loss = torch.nn.MSELoss(reduction="sum")
        self.args
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output ,ret= model(**sample['net_input'])
        distillation_output, bert_output = ret['distillation_out'], ret['bert_encoder_out']

        loss= self.compute_loss(model, net_output, sample, distillation_output, bert_output)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        #import pdb
        #pdb.set_trace()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    # def compute_loss(self, model, net_output, sample, reduce=True):
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1)
    #     loss = F.nll_loss(
    #         lprobs,
    #         target,
    #         ignore_index=self.padding_idx,
    #         reduction='sum' if reduce else 'none',
    #     )
    #     return loss, loss

    def compute_loss(self, model, net_output, sample, distillation_output, bert_output, alpha=0.9, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        #TODO : if
        f1 = torch.mean(distillation_output, dim=0, keepdim=True)
        f2 = torch.mean(bert_output['bert_encoder_out'], dim=0, keepdim=True)
        loss_kd = self.MSE_loss(f1.float(), f2.float())
        return loss * alpha + loss_kd * (1. - alpha)
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
