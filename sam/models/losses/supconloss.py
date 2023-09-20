"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=2., contrast_mode='all',
                 base_temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask = mask[(labels > 0)[:, 0], :]
        log_prob = log_prob[(labels > 0)[:, 0], :]
        uses = mask.sum(1) > 0
        mask = mask[uses, :]
        log_prob = log_prob[uses, :]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, mean_log_prob_pos.shape[0]).mean()

        return loss


@LOSSES.register_module()
class SupConTopKLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=1., contrast_mode='all',
                 base_temperature=0.5):
        super(SupConTopKLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_feature = anchor_feature[labels.view(-1) > 0, :]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits = logits * 10

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )

        neg_mask = 1 - mask
        mask = mask * logits_mask

        mask = mask[labels.view(-1) > 0, :]
        neg_mask = neg_mask[labels.view(-1) > 0, :]
        logits_mask = logits_mask[labels.view(-1) > 0, :]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        neg_exp_logits = exp_logits * neg_mask
        v, ind = neg_exp_logits.topk(200)
        mask_topk = torch.zeros_like(mask).scatter_(1, ind, 1)
        all_mask = mask + mask_topk
        all_use_exp_logits = exp_logits * all_mask
        log_prob = logits - torch.log(all_use_exp_logits.sum(1, keepdim=True))
        # mask = mask[(labels>0)[:,0],:]
        # log_prob = log_prob[(labels>0)[:,0],:]
        uses = mask.sum(1) > 0
        mask = mask[uses, :]
        log_prob = log_prob[uses, :]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, mean_log_prob_pos.shape[0]).mean()

        return loss


@LOSSES.register_module()
class SupConNegLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=1.0, contrast_mode='all',
                 base_temperature=0.5):
        super(SupConNegLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.zeros_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            1
        )
        # mask = mask * logits_mask

        mask = 1 - mask
        mask = mask + logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * mask
        # v,ind = exp_logits.topk(200)
        # mask_topk = torch.zeros_like(mask).scatter_(1,ind, 1)
        # exp_logits = exp_logits * mask_topk
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(20, 20))
        # plt.imshow(mask_topk.data.cpu())
        # plt.show()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask = mask[(labels>0)[:,0],:]
        # log_prob = log_prob[(labels>0)[:,0],:]
        # uses = mask.sum(1)>0
        # mask = mask[uses,:]
        # log_prob = log_prob[uses,:]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (logits_mask * log_prob).sum(1) / logits_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, mean_log_prob_pos.shape[0]).mean()

        return loss


@LOSSES.register_module()
class SupConMeanLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.01, contrast_mode='all',
                 base_temperature=1.):
        super(SupConMeanLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, prototypes, prototypes_labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        prototypes_labels = prototypes_labels.view(-1, 1).to(labels.device)
        mask_p_f = torch.eq(prototypes_labels, labels.T).float().to(features.device)

        features = features.squeeze(1)
        # compute logits
        prototypes_dot_features = torch.div(
            torch.matmul(prototypes, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(prototypes_dot_features, dim=1, keepdim=True)
        logits = prototypes_dot_features - logits_max.max().detach()
        # logits = prototypes_dot_features

        # compute log_prob
        exp_logits = torch.exp(logits)
        # v,ind = exp_logits.topk(200)
        # mask_topk = torch.zeros_like(mask).scatter_(1,ind, 1)
        # exp_logits = exp_logits * mask_topk
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(20, 20))
        # plt.imshow(mask_topk.data.cpu())
        # plt.show()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask = mask[(labels>0)[:,0],:]
        # log_prob = log_prob[(labels>0)[:,0],:]
        uses = mask_p_f.sum(1) > 0
        mask_p_f = mask_p_f[uses, :]
        log_prob = log_prob[uses, :]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_p_f * log_prob).sum(1) / mask_p_f.sum(1)

        # loss
        loss = - self.base_temperature * mean_log_prob_pos
        loss = loss.view(1, mean_log_prob_pos.shape[0]).mean()
        return loss


@LOSSES.register_module()
class SupConMeanLossCol(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=1.0):
        super(SupConMeanLossCol, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, prototypes, prototypes_labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.view(-1, 1).to(features.device)
        prototypes_labels = prototypes_labels.view(-1, 1).to(features.device)
        # mask_p_f = torch.eq(prototypes_labels,labels.T).float().to(features.device)

        mask_f_p = torch.eq(labels, prototypes_labels.T).float().to(features.device)

        features = features.squeeze(1)
        # compute logits
        features_dot_prototypes = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        # prototypes_dot_features = torch.div(
        #     torch.matmul(prototypes,features.T),
        #     self.temperature)
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = features_dot_prototypes

        # compute log_prob
        exp_logits = torch.exp(logits)
        # v,ind = exp_logits.topk(200)
        # mask_topk = torch.zeros_like(mask).scatter_(1,ind, 1)
        # exp_logits = exp_logits * mask_topk
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(20, 20))
        # plt.imshow(mask_topk.data.cpu())
        # plt.show()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask = mask[(labels>0)[:,0],:]
        # log_prob = log_prob[(labels>0)[:,0],:]
        # uses = mask.sum(1)>0
        # mask = mask[uses,:]
        # log_prob = log_prob[uses,:]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_f_p * log_prob).sum(1) / mask_f_p.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(1, mean_log_prob_pos.shape[0]).mean()

        return loss
