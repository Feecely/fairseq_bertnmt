# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class NewMaskFillDistillationLossCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    kd_alpha: float = field(
        default=0.9,
        metadata={"help": "..."},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "new_mask_fill_distillation_loss", dataclass=NewMaskFillDistillationLossCriterionConfig
)
class NewMaskFillDistillationLossCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kd_alpha=0.9,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.MSE_loss = torch.nn.MSELoss(reduce=False, reduction="sum")
        self.alpha = kd_alpha
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, ret = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        BERT_encoder_mapping, BART_encoder_mapping = ret['BERT_encoder_mapping'], ret['BART_encoder_mapping']

        mask_bert_out, mask_encoder_out = ret['mask_bert_out'], ret['mask_encoder_out']
        # import pdb
        # pdb.set_trace()
        if BERT_encoder_mapping is not None:
            mask_encoder_out = self.merge_encoder(mask_bert_out, mask_encoder_out, BERT_encoder_mapping, -1)
        mask_loss = ret['mask_loss']
        loss_kd = self.MSE_loss(mask_bert_out, mask_encoder_out)
        loss_kd = torch.mean(loss_kd, dim=-1)
        loss_kd = loss_kd.sum()

        fill_bart_out, fill_encoder_out = ret['fill_bart_out'], ret['fill_encoder_out']
        #import pdb; pdb.set_trace()
        if BART_encoder_mapping is not None:
            fill_encoder_out = self.merge_encoder(fill_bart_out, fill_encoder_out, BART_encoder_mapping, -1)
        fill_loss = ret['fill_loss']

        fill_loss_kd = self.MSE_loss(fill_bart_out, fill_encoder_out)
        fill_loss_kd = torch.mean(fill_loss_kd, dim=-1).sum()
        import pdb; pdb.set_trace()
        loss = loss * self.alpha + mask_loss + fill_loss + (fill_loss_kd + loss_kd) * (1. - self.alpha)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    # def merge_bert_encoder(self, bert_out, encoder_out, dict):
    #     tmp = torch.zeros(bert_out.shape).cuda()
    #     for line in range(len(dict)):
    #         tmp[line][0] = encoder_out[line][0]
    #         size = 0
    #         for num in range(len(tmp[line])):
    #             if num < len(dict[line]) and dict[line][num] != 0:
    #                     tmp[line][dict[line][num]] = tmp[line][dict[line][num]] + encoder_out[line][dict[line][num]]
    #                     size = dict[line][num]
    #         for num in range(size + 1, len(tmp[line])):
    #             tmp[line][num] = tmp[line][num] + encoder_out[line][num]
    #     return tmp

    def merge_encoder(self, bert_out, encoder_out, dict, pad):

        remove_dim = bert_out.shape[1]
        dict = (dict != pad) * dict + (dict == pad) * remove_dim # B * T
        import pdb; pdb.set_trace()
        merge_shape = encoder_out.shape
        merge_shape[1] = merge_shape[1] + 1
        merge_encoder_out = torch.zeros(merge_shape).cuda().index_add_(1, dict, encoder_out)
        return merge_encoder_out[:, : -1]

    # def merge_bart_encoder(self, bert_out, encoder_out, dict):
    #     tmp = torch.zeros(bert_out.shape).cuda()
    #     for line in range(len(dict)):
    #         tmp[line][1] = encoder_out[line][1]
    #         for num in range(len(tmp[line])):
    #             if num < len(dict[line]) and dict[line][num] != 1:
    #                     tmp[line][dict[line][num]] = tmp[line][dict[line][num]] + encoder_out[line][dict[line][num]]
    #                     size = dict[line][num]
    #         for num in range(size + 1, len(tmp[line])):
    #             tmp[line][num] = tmp[line][num] + encoder_out[line][num]
    #     return tmp

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
