# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_

# from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.loss import RegLoss, BPRLoss


class CpRec(nn.Module):
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

    def __init__(self, args): #config, dataset
        super(CpRec, self).__init__()#config, dataset

        # load parameters info
        self.embedding_size = args.embedding_size #config['embedding_size']
        self.residual_channels = args.embedding_size #config['embedding_size']
        self.block_num = args.block_num #config['block_num']
        self.dilations = args.dilations * self.block_num #config['dilations'] * self.block_num
        self.kernel_size = args.kernel_size #config['kernel_size']
        self.output_dim = args.num_items
        self.pad_token = args.pad_token
        self.cp_type = args.cp_type

        # self.reg_weight = config['reg_weight']
        # self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.output_dim+1, self.embedding_size, padding_idx=self.pad_token)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        if args.cp_type == 'cro_layer':
            global cro_conv_weight, cro_conv_bias
            cro_conv_weight = nn.Parameter(torch.randn(self.residual_channels, self.residual_channels, 1, self.kernel_size)).to(args.device)
            cro_conv_bias = nn.Parameter(torch.randn(self.residual_channels, )).to(args.device)
            # rb = [
            #     ResidualBlock_cro_layer_b(
            #         self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            #     ) for dilation in self.dilations
            # ]
            rb = [ResidualBlock_cro_layer_b(args, dilation=dilation) for dilation in self.dilations]
        elif args.cp_type == 'adj_layer':
            rb = [ResidualBlock_cro_layer_b(args, dilation=dilation) for dilation in self.dilations]
        elif args.cp_type == 'cro_block':
            # global cro_conv_weight_0, cro_conv_bias_0, cro_conv_weight_1, cro_conv_bias_1
            # cro_conv_weight_0 = nn.Parameter(torch.randn(self.residual_channels, self.residual_channels, 1, self.kernel_size)).to(args.device)
            # cro_conv_bias_0 = nn.Parameter(torch.randn(self.residual_channels, )).to(args.device)
            # cro_conv_weight_1 = nn.Parameter(torch.randn(self.residual_channels, self.residual_channels, 1, self.kernel_size)).to(args.device)
            # cro_conv_bias_1 = nn.Parameter(torch.randn(self.residual_channels, )).to(args.device)
            # rb = [
            #     ResidualBlock_cro_block_b(
            #         self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            #     ) for dilation in self.dilations
            # ]
            self.conv_param1 = conv_param(args, self.residual_channels, self.kernel_size)
            self.conv_param2 = conv_param(args, self.residual_channels, self.kernel_size)
            rb = [ResidualBlock_cro_block_b(args, self.conv_param1, self.conv_param2, dilation=dilation) for dilation in self.dilations]
        elif args.cp_type == 'adj_block':
            i = 1
            rb = []
            # global adj_conv_weight_0, adj_conv_bias_0, adj_conv_weight_1, adj_conv_bias_1
            for dilation in self.dilations:
                if i%2 == 1:
                    self.conv_param1 = conv_param(args, self.residual_channels, self.kernel_size)
                    self.conv_param2 = conv_param(args, self.residual_channels, self.kernel_size)
                    # adj_conv_weight_0 = nn.Parameter(torch.randn(self.residual_channels, self.residual_channels, 1, self.kernel_size)).to(args.device)
                    # adj_conv_bias_0 = nn.Parameter(torch.randn(self.residual_channels, )).to(args.device)
                    # adj_conv_weight_1 = nn.Parameter(torch.randn(self.residual_channels, self.residual_channels, 1, self.kernel_size)).to(args.device)
                    # adj_conv_bias_1 = nn.Parameter(torch.randn(self.residual_channels, )).to(args.device)
                # block = ResidualBlock_adj_block_b(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation)
                block = ResidualBlock_adj_block_b(args, self.conv_param1, self.conv_param2, dilation=dilation)
                rb.append(block)
                i += 1
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.output_dim+1)

        # parameters initialization
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / (self.output_dim+1))
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, item_seq):
        # print("--------", item_seq.max())
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        # hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        seq_output = self.final_layer(dilate_outputs)  # [batch_size, embedding_size]hidden
        return seq_output

# class Conv(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
#         # self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
#
#     def forward(self, x):
#         out = self.conv(x)
#         return out

class conv_param():
    def __init__(self, args, residual_channels, kernel_size):
        self.weight = nn.Parameter(torch.randn(residual_channels, residual_channels, 1, kernel_size)).to(args.device)
        self.bias = nn.Parameter(torch.randn(residual_channels, )).to(args.device)

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


class ResidualBlock_adj_layer_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, args, dilation=None): #in_channel, out_channel, kernel_size=3,
        super(ResidualBlock_adj_layer_b, self).__init__()
        self.in_channel = args.embedding_size
        self.out_channel = args.embedding_size
        self.kernel_size = args.kernel_size

        self.conv_weight = nn.Parameter(torch.randn(self.in_channel, self.out_channel, 1, self.kernel_size)).to(args.device)
        self.conv_bias = nn.Parameter(torch.randn(self.out_channel,)).to(args.device)
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        # shape = self.conv1._parameters
        self.ln1 = nn.LayerNorm(self.out_channel, eps=1e-8)

        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(self.out_channel, eps=1e-8)
        self.dilation = dilation

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = F.conv2d(x_pad, weight=self.conv_weight, bias=self.conv_bias, stride=1, padding=0, dilation=self.dilation, groups=1)
        out = out.squeeze(2).permute(0, 2, 1)
        # out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = F.conv2d(out_pad, weight=self.conv_weight, bias=self.conv_bias, stride=1, padding=0, dilation=self.dilation * 2, groups=1)
        out2 = out2.squeeze(2).permute(0, 2, 1)
        # out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

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

class ResidualBlock_cro_layer_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, args, dilation=None):#in_channel, out_channel, kernel_size=3,
        super(ResidualBlock_cro_layer_b, self).__init__()
        self.in_channel = args.embedding_size
        self.out_channel = args.embedding_size
        self.kernel_size = args.kernel_size
        # self.cro_conv_weight = cro_conv_weight
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(self.out_channel, eps=1e-8)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(self.out_channel, eps=1e-8)
        self.dilation = dilation

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        # out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F.conv2d(x_pad, weight=cro_conv_weight, bias=cro_conv_bias, stride=1, padding=0, dilation=self.dilation,
                       groups=1)
        out = out.squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        # out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.conv2d(out_pad, weight=cro_conv_weight, bias=self.cro_conv_bias, stride=1, padding=0,
                        dilation=self.dilation * 2, groups=1)
        out2 = out2.squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

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

class ResidualBlock_adj_block_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, args, conv_param1, conv_param2, dilation=None): #in_channel, out_channel, kernel_size=3,
        super(ResidualBlock_adj_block_b, self).__init__()
        self.in_channel = args.embedding_size
        self.out_channel = args.embedding_size
        self.kernel_size = args.kernel_size
        # adj_conv_weight_0 = nn.Parameter(torch.randn(self.in_channel, self.out_channel, 1, self.kernel_size))
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        # shape = self.conv1._parameters
        self.ln1 = nn.LayerNorm(self.out_channel, eps=1e-8)

        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(self.out_channel, eps=1e-8)
        self.conv_param1 = conv_param1
        self.conv_param2 = conv_param2
        self.dilation = dilation
        # self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = F.conv2d(x_pad, weight=self.conv_param1.weight, bias=self.conv_param1.bias, stride=1, padding=0, dilation=self.dilation, groups=1)
        out = out.squeeze(2).permute(0, 2, 1)
        # out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = F.conv2d(out_pad, weight=self.conv_param2.weight, bias=self.conv_param2.bias, stride=1, padding=0, dilation=self.dilation * 2, groups=1)
        out2 = out2.squeeze(2).permute(0, 2, 1)
        # out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

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

class ResidualBlock_cro_block_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, args, conv_param1, conv_param2, dilation=None): #in_channel, out_channel, kernel_size=3,
        super(ResidualBlock_cro_block_b, self).__init__()
        self.in_channel = args.embedding_size
        self.out_channel = args.embedding_size
        self.kernel_size = args.kernel_size
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        # self.conv1 = Conv(self.in_channel, self.out_channel, kernel_size=self.kernel_size, dilation=dilation)
        self.ln1 = nn.LayerNorm(self.out_channel, eps=1e-8)
        self.conv_param1 = conv_param1
        self.conv_param2 = conv_param2

        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        # self.conv2 = Conv(self.in_channel, self.out_channel, kernel_size=self.kernel_size, dilation=dilation*2)
        self.ln2 = nn.LayerNorm(self.out_channel, eps=1e-8)
        self.dilation = dilation
        # self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = F.conv2d(x_pad, weight=self.conv_param1.weight, bias=self.conv_param1.bias, stride=1, padding=0, dilation=self.dilation, groups=1)
        # out = self.conv1(x_pad)
        out = out.squeeze(2).permute(0, 2, 1)
        # out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = F.conv2d(out_pad, weight=self.conv_param2.weight, bias=self.conv_param2.bias, stride=1, padding=0, dilation=self.dilation * 2, groups=1)
        # out2 = self.conv2(out_pad)
        out2 = out2.squeeze(2).permute(0, 2, 1)
        # out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

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