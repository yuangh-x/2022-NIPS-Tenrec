# -*- coding: utf-8 -*-
'''
Reference:
    [1]Lei Chen et al. A user-adaptive layer selection framework for very deep sequential recommender models. In Proceedings of the AAAI Conference on
    Artificial Intelligence, volume 35, pages 3984–3991, 2021.
'''
import numpy as np
import torch
import time
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_

class SkipRec(nn.Module):
    r"""The network architecture of the NextItNet model is formed of a stack of holed convolutional layers, which can
    efficiently increase the receptive fields without relying on the pooling operation.
    Also residual block structure is used to ease the optimization for much deeper networks.

    Note:
        As paper said, for comparison purpose, we only predict the next one item in our evaluation,
        and then stop the generating process. Although the number of parameters in residual block (a) is less
        than it in residual block (b), the performance of b is better than a.
        So in our model, we use residual block (b).
        In addition, when dilations is not equal to 1, the training may be slow. To  speed up the efficiency, please set the parameters "reproducibility" False.
    """

    def __init__(self, args):
        super(SkipRec, self).__init__()

        # load parameters info
        self.embedding_size = args.embedding_size
        self.residual_channels = args.embedding_size
        self.block_num = args.block_num
        self.dilations = args.dilations * self.block_num
        self.kernel_size = args.kernel_size
        self.output_dim = args.num_items
        self.pad_token = args.pad_token
        self.all_time = 0

        # define layers and loss
        self.item_embedding = nn.Embedding(self.output_dim+1, self.embedding_size, padding_idx=self.pad_token)

        # residual blocks
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.output_dim+1)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / (self.output_dim+1))
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
    def forward(self, item_seq, policy_action):
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_input = item_seq_emb
        for layer_id, block in enumerate(self.residual_blocks):
            layer_input = dilate_input
            action_mask = policy_action[:, layer_id].reshape([-1, 1, 1])
            layer_output = block(dilate_input)

            dilate_input = layer_output * action_mask + layer_input * (1 - action_mask)
        seq_output = self.final_layer(dilate_input) #dilate_outputs  # [batch_size, embedding_size]hidden
        return seq_output

    def predict(self, item_seq, policy_action):
        # note: batch_size = 1
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_input = item_seq_emb
        policy_action = policy_action.squeeze(0)
        layer_input = dilate_input
        since_time = time.time()
        for layer_id, block in enumerate(self.residual_blocks):
            if policy_action[layer_id].eq(1):
                layer_input = block(layer_input)
        one_time = time.time() - since_time
        self.all_time += one_time
        seq_output = self.final_layer(layer_input)  # dilate_outputs  # [batch_size, embedding_size]hidden
        return seq_output

class ResidualBlock_a(nn.Module):
    r"""
    Residual block (a) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_a, self).__init__()

        half_channel = out_channel // 2
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv1 = nn.Conv2d(in_channel, half_channel, kernel_size=(1, 1), padding=0)

        self.ln2 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(half_channel, half_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)

        self.ln3 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv3 = nn.Conv2d(half_channel, out_channel, kernel_size=(1, 1), padding=0)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]

        out = F.relu(self.ln1(x))
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)

        out2 = F.relu(self.ln2(out))
        out2 = self.conv_pad(out2, self.dilation)
        out2 = self.conv2(out2).squeeze(2).permute(0, 2, 1)

        out3 = F.relu(self.ln3(out2))
        out3 = out3.permute(0, 2, 1).unsqueeze(2)
        out3 = self.conv3(out3).squeeze(2).permute(0, 2, 1)
        return out3 + x

    def conv_pad(self, x, dilation):  # x: [batch_size, seq_len, embed_size]
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        # padding operation  args：(left,right,top,bottom)
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.rez = nn.Parameter(torch.FloatTensor([1]))

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 * self.rez + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad

class PolicyNetGumbel(nn.Module):
    def __init__(self, args):
        super(PolicyNetGumbel, self).__init__()
        self.device = args.device
        self.temp = args.temp
        self.embedding_size = args.embedding_size
        self.residual_channels = args.embedding_size
        self.block_num = args.block_num
        self.dilations = args.dilations
        self.kernel_size = args.kernel_size
        self.output_dim = args.num_items
        self.pad_token = args.pad_token
        self.action_num = len(self.dilations * self.block_num)

        self.item_embedding = nn.Embedding(self.output_dim + 1, self.embedding_size, padding_idx=self.pad_token)

        # residual blocks
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.action_num * 2)

    def forward(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        seq_output = self.final_layer(dilate_outputs)  # [batch_size, embedding_size]hidden
        seq_output = seq_output.mean(1)
        seq_output = seq_output.reshape([-1, self.action_num, 2])
        seq_output = torch.sigmoid(seq_output)
        seq_output = F.softmax(seq_output, dim=-1)
        action = self.gumbel_softmax(seq_output, temp=self.temp, hard=True)
        action_predict = action[:, :, 0]
        return action_predict

    def gumbel_softmax(self, logits, temp=10, hard=False):
        gumbel_softmax_sample = logits + self.sample_gumbel(logits.shape)
        y = F.softmax(gumbel_softmax_sample / temp, dim=-1)

        if hard:
            y_hard = torch.eq(y, torch.max(y, -1, keepdim=True)[0]).to(y.dtype)
            y1 = y_hard - y
            y1 = y1.detach()
            y = y1 + y
        return y

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.Tensor(shape).uniform_(0, 1).to(self.device)
        return -torch.log(-torch.log(u + eps) + eps)