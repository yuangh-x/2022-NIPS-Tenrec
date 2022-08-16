import torch.nn as nn

class DNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.block_num = args.block_num
        self.num_items = args.num_items
        self.pad_token = args.pad_token
        self.dropout = args.dropout


        self.embedding = nn.Embedding(self.num_items+1, self.embedding_size, padding_idx=self.pad_token)
        self.blocks = nn.ModuleList([Linear_Block(self.hidden_size, self.dropout) for _ in range(self.block_num)])
        self.out = nn.Linear(self.hidden_size, args.num_labels)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)




class Linear_Block(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Linear_Block, self).__init__()
        self.hidden_size = hidden_size

        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.linear(x)
        out = self.ln(out)
        out = self.dropout(out)

        return out + x


