from torch import nn
'''
Reference:
    [1]Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.
Reference:
    https://github.com/RUCAIBox/RecBole
'''
class GRU4Rec(nn.Module):
    r"""
    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__()

        self.embedding_size = args.embedding_size
        self.vocab_size = args.num_items + 1
        self.n_layers = args.block_num
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

        # define layers and loss
        self.item_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

