'''
Reference:
    [1]Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems.
    Computer, 42(8):30–37, 2009.
Reference:
    https://github.com/recsys-benchmark/DaisyRec-v2.0
'''
import torch
import torch.nn as nn
import numpy as np
from model.cf.AbstractRecommender import GeneralRecommender

class MF(GeneralRecommender):
    def __init__(self, args):
        """
        Matrix Factorization Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(MF, self).__init__(args)

        self.epochs = args.epochs
        self.lr = args.lr
        self.reg_1 = args.reg_1
        self.reg_2 = args.reg_2
        self.topk = args.k
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.dropout = args.dropout
        self.factor_num = args.factor_num

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num, self.factor_num)

        self.loss_type = args.loss_type
        self.optimizer = args.optimizer if args.optimizer != 'default' else 'sgd'
        self.initializer = args.init_method if args.init_method != 'default' else 'normal'
        self.early_stop = args.early_stop
        self.apply(self._init_weight)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        pred = (embed_user * embed_item).sum(dim=-1)


        return pred #.view(-1)

    def calc_loss(self, batch):
        user = batch[0].to(self.device).long()
        pos_item = batch[1].to(self.device).long()
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)

            # add regularization term
            loss += self.reg_1 * (self.embed_item(pos_item).norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device).long()
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            # add regularization term
            loss += self.reg_1 * (self.embed_item(pos_item).norm(p=1) + self.embed_item(neg_item).norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).norm() + self.embed_item(neg_item).norm())
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        # add regularization term
        loss += self.reg_1 * (self.embed_user(user).norm(p=1))
        loss += self.reg_2 * (self.embed_user(user).norm())

        return loss

    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        pred = self.forward(u, i).cpu().item()

        return pred

    def rank(self, test_loader):
        rec_ids = torch.tensor([], device=self.device)
        self.eval()
        with torch.no_grad():
            for us in test_loader:
                rank_list = self.full_rank(us)
                rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy().astype(np.int)

    def full_rank(self, u):
        u = u.to(self.device)

        user_emb = self.embed_user(u)
        items_emb = self.embed_item.weight
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0)) #  # (item_num,)
        return torch.argsort(scores, descending=True)[:, :self.topk]#.cpu().numpy()

