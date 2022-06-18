from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

AdaptiveSoftmaxOutput = namedtuple('AdaptiveSoftmaxOutput', ['output', 'loss'])


class AdaptiveTail(nn.Module):
    def __init__(self, ndim, ntoken, cutoffs, div_value=4):
        super(AdaptiveTail, self).__init__()
        self.div_value = div_value
        self.ndim = ndim
        self.cutoffs = cutoffs + [ntoken]
        self.tail_clusters = nn.ModuleList()
        for i, l_bound in enumerate(self.cutoffs[:-1]):
            cluster_size = self.cutoffs[i + 1] - l_bound
            self.tail_clusters.append(
                nn.Sequential(
                    nn.Embedding(cluster_size, ndim // (div_value ** (i + 1))),
                    nn.Linear(ndim // (div_value ** (i + 1)), self.ndim, bias=False)
                )
            )

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
            elif hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, inputs, cluster, softmax=True):
        if softmax:
            outputs = F.linear(inputs, self.tail_clusters[cluster][1].weight.T)
            return F.linear(outputs, self.tail_clusters[cluster][0].weight)
        else:
            return self.tail_clusters[cluster](inputs)


class AdaptiveSoftmax(nn.Module):
    def __init__(self, ndim, ntoken, cutoffs, div_value=4, shared_tail=None):
        super(AdaptiveSoftmax, self).__init__()
        self.div_value = div_value
        self.ndim = ndim
        self.cutoffs = cutoffs + [ntoken]
        self.head_size = self.cutoffs[0] + len(self.cutoffs) - 1

        self.head_cluster = nn.Linear(self.ndim, self.head_size, bias=False)
        nn.init.xavier_uniform_(self.head_cluster.weight)

        if shared_tail is not None:
            self.tail_clusters = shared_tail
        else:
            self.tail_clusters = AdaptiveTail(ndim, ntoken, cutoffs, div_value)

    def map_target_to_cluster(self, targets):
        cluster_targets = []
        head_targets = targets.clone()
        for i in range(len(self.cutoffs) - 1):
            l_bound = self.cutoffs[i]
            u_bound = self.cutoffs[i + 1]
            targets_in_range = targets.ge(l_bound).logical_and(targets.lt(u_bound))
            targets_in_range = targets_in_range.nonzero().squeeze(dim=1)
            cluster_targets.append(targets_in_range)
            head_targets[targets_in_range] = self.cutoffs[0] + i
        return cluster_targets, head_targets

    def forward(self, inputs, targets):
        outputs = inputs.new_zeros(targets.size(0))

        cluster_targets, head_targets = self.map_target_to_cluster(targets)
        head_output = self.head_cluster(inputs)
        head_output = head_output.log_softmax(dim=1)
        head_output = head_output.gather(1, head_targets.unsqueeze(1))
        outputs += head_output.squeeze()

        for i, ids in enumerate(cluster_targets):
            if len(ids) == 0:  # no targets for this cluster
                continue
            cluster_outputs = self.tail_clusters(inputs[ids], i, softmax=True)
            cluster_outputs = cluster_outputs.log_softmax(dim=1)
            relative_targets = targets[ids] - self.cutoffs[i]
            cluster_outputs = cluster_outputs.gather(1, relative_targets.unsqueeze(1))
            outputs[ids] += cluster_outputs.squeeze()

        loss = (-outputs).mean()
        return AdaptiveSoftmaxOutput(outputs, loss)


class AdaptiveInput(nn.Module):
    def __init__(self, ndim, ntoken, cutoffs, div_value=4, shared_tail=None):
        super(AdaptiveInput, self).__init__()
        self.div_value = div_value
        self.ndim = ndim
        self.cutoffs = cutoffs + [ntoken]
        self.head_size = self.cutoffs[0] + len(self.cutoffs) - 1

        self.head_cluster = nn.Sequential(
            nn.Embedding(self.cutoffs[0], self.ndim),
            nn.Linear(self.ndim, self.ndim)
        )
        nn.init.normal_(self.head_cluster[0].weight, mean=0, std=self.head_cluster[0].weight.shape[1] ** -0.5)
        nn.init.xavier_uniform_(self.head_cluster[1].weight)
        if shared_tail is not None:
            self.tail_clusters = shared_tail
        else:
            self.tail_clusters = AdaptiveTail(ndim, ntoken, cutoffs, div_value)

    def forward(self, inputs):
        outputs = inputs.new_zeros(inputs.shape + (self.ndim,), dtype=torch.float)
        cutoffs = [0] + self.cutoffs
        for i in range(len(cutoffs) - 1):
            l_bound = cutoffs[i]
            u_bound = cutoffs[i + 1]
            cluster_mask = inputs.ge(l_bound).logical_and(inputs.lt(u_bound))
            cluster_inputs = inputs[cluster_mask] - cutoffs[i]
            if i == 0:
                cluster_output = self.head_cluster(cluster_inputs)
            else:
                cluster_output = self.tail_clusters(cluster_inputs, i - 1, softmax=False)
            outputs[cluster_mask] = cluster_output
        return outputs

