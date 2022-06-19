import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, args, policy=False):
        super().__init__()
        max_len = args.max_len
        num_items = args.num_items
        if policy:
            n_layers = int(args.block_num / 4)
        else:
            n_layers = args.block_num
        heads = args.num_heads
        vocab_size = num_items + 1
        self.hidden = args.hidden_size
        dropout = args.dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, heads, self.hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass

class SAS4infaccModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)
        self.all_time = 0

    def forward(self, x, policy_action):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        dilate_input = self.bert.embedding(x)
        for layer_id, block in enumerate(self.bert.transformer_blocks):
            layer_input = dilate_input
            action_mask = policy_action[:, layer_id].reshape([-1, 1, 1])
            layer_output = block(dilate_input, mask)
            dilate_input = layer_output * action_mask + layer_input * (1 - action_mask)

        return self.out(dilate_input)

    def predict(self, x, policy_action):
        # note: batch_size = 1
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        dilate_input = self.bert.embedding(x)
        # Residual locks
        policy_action = policy_action.squeeze(0)
        layer_input = dilate_input
        since_time = time.time()
        for layer_id, block in enumerate(self.bert.transformer_blocks):
            if policy_action[layer_id].eq(1):
                layer_input = block(layer_input, mask)
        one_time = time.time() - since_time
        self.all_time += one_time
        seq_output = self.out(layer_input)  # dilate_outputs  # [batch_size, embedding_size]hidden
        return seq_output


class SAS_PolicyNetGumbel(nn.Module):
    def __init__(self, args):
        super(SAS_PolicyNetGumbel, self).__init__()
        self.device = args.device
        self.temp = args.temp
        self.action_num = args.block_num
        self.bert = BERT(args, policy=True)
        self.out = nn.Linear(self.bert.hidden, self.action_num * 2)

    def forward(self, x):
        x = self.bert(x)
        seq_output = self.out(x)  # [batch_size, embedding_size]hidden
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