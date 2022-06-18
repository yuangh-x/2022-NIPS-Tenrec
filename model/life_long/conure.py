# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_


class Conure(nn.Module):

    def __init__(self, args): #config, dataset
        super(Conure, self).__init__()#config, dataset

        # load parameters info
        self.embedding_size = args.embedding_size #config['embedding_size']
        self.residual_channels = args.embedding_size #config['embedding_size']
        self.block_num = args.block_num #config['block_num']
        self.dilations = args.dilations * self.block_num #config['dilations'] * self.block_num
        self.kernel_size = args.kernel_size #config['kernel_size']
        self.output_dim = args.num_items
        self.vocab_size = args.num_embedding + 1
        self.pad_token = args.pad_token
        self.times = args.task
        self.output_dim1 = args.task1_out
        self.output_dim2 = args.task2_out
        self.output_dim3 = args.task3_out
        self.output_dim4 = args.task4_out

        # self.reg_weight = config['reg_weight']
        # self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_token)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        # self.final_layer = nn.Linear(self.residual_channels, self.output_dim+1)
        self.final_layer1 = nn.Linear(self.residual_channels, self.output_dim1 + 1)
        self.final_layer2 = nn.Linear(self.residual_channels, self.output_dim2 + 1)
        self.final_layer3 = nn.Linear(self.residual_channels, self.output_dim3 + 1)
        self.final_layer4 = nn.Linear(self.residual_channels, self.output_dim4 + 1)

        # if self.loss_type == 'BPR':
        #     self.loss_fct = BPRLoss()
        # elif self.loss_type == 'CE':
        #     self.loss_fct = nn.CrossEntropyLoss()
        # else:
        #     raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        # self.reg_loss = RegLoss()

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

    def forward(self, item_seq):#, pos, neg
        # print("--------", item_seq.max())
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        # hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        # seq_output = self.final_layer(dilate_outputs)  # [batch_size, embedding_size]hidden
        if self.times == 0:
            seq_output = self.final_layer1(dilate_outputs)  # [batch_size, embedding_size]hidden
        elif self.times == 1:
            seq_output = self.final_layer2(dilate_outputs)
        elif self.times == 2:
            seq_output = self.final_layer3(dilate_outputs)
        else:
            seq_output = self.final_layer4(dilate_outputs)
        # pos_emb = self.item_embedding(pos)
        # neg_emb = self.item_embedding(neg)
        # pos_logit = (dilate_outputs * pos_emb).mean(-1)
        # neg_logit = (dilate_outputs * neg_emb).mean(-1)
        return seq_output#pos_logit, neg_logit

    def predict(self, log_seqs, item):  # for inference
        item_seq_emb = self.item_embedding(log_seqs)  # [batch_size, seq_len, embed_size]
        # Residual locks
        log_feats = self.residual_blocks(item_seq_emb)

        item_embs = self.item_embedding(item)  # (U, I, C)
        logits = log_feats.matmul(item_embs.transpose(1, 2))  # .squeeze(-1)
        logits = logits.mean(1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # scores# preds # (U, I)

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

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
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