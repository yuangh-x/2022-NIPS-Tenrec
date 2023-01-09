'''
Reference:
    https://github.com/recsys-benchmark/DaisyRec-v2.0
'''
import torch.nn as nn
import torch.optim as optim
from metrics import *

class AbstractRecommender(nn.Module):
    def __init__(self):
        super(AbstractRecommender, self).__init__()
        self.optimizer = None
        self.initializer = None
        self.loss_type = None
        self.lr = 0.01
        self.logger = None

        self.initializer_param_config = {
            'normal': {'mean': 0.0, 'std': 0.01},
            'uniform': {'a': 0.0, 'b': 1.0},
            'xavier_normal': {'gain': 1.0},
            'xavier_uniform': {'gain': 1.0}
        }

        self.initializer_config = {
            'normal': nn.init.normal_,
            'uniform': nn.init.uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_
        }

    def calc_loss(self, batch):
        raise NotImplementedError

    def fit(self, train_loader, val_loader):
        raise NotImplementedError

    def rank(self, test_loader):
        raise NotImplementedError

    def full_rank(self, u):
        raise NotImplementedError

    def predict(self, u, i):
        raise NotImplementedError

    def _build_optimizer(self, **kwargs):
        params = self.parameters()
        learner = kwargs.pop('optimizer', self.optimizer)
        learning_rate = kwargs.pop('lr', self.lr)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            self.logger.info('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)

        return optimizer

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            self.initializer_config[self.initializer](m.weight, **self.initializer_param_config[self.initializer])
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            self.initializer_config[self.initializer](m.weight, **self.initializer_param_config[self.initializer])
        else:
            pass

    def _build_criterion(self, loss_type):
        if loss_type.upper() == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='sum')
        elif loss_type.upper() == 'SL':
            criterion = nn.MSELoss(reduction='sum')
        elif loss_type.upper() == 'BPR':
            criterion = BPRLoss()
        elif loss_type.upper() == 'HL':
            criterion = HingeLoss()
        elif loss_type.upper() == 'TL':
            criterion = TOP1Loss()
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}...')

        return criterion


class GeneralRecommender(AbstractRecommender):
    def __init__(self, args):
        super(GeneralRecommender, self).__init__()
        self.args = args
        # os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.logger = config['logger']
        self.device = args.device

    def fit(self, train_loader, val_loader):
        self.to(self.device)
        optimizer = self._build_optimizer(optimizer=self.optimizer, lr=self.lr)
        if self.loss_type is not None:
            self.criterion = self._build_criterion(self.loss_type)
        else:
            self.criterion = None

        last_loss = 0.
        best_metrics = 0
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # pbar = train_loader
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            # print(f'[Epoch {epoch:03d}]')
            for batch in pbar:
                self.zero_grad()
                loss = self.calc_loss(batch)

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                current_loss += loss.item()
            pbar.set_postfix(loss=current_loss)

            preds = self.rank(val_loader)
            ndcg = NDCG(self.args.val_ur, preds, self.args.val_u)
            recall = Recall(self.args.val_ur, preds, self.args.val_u)
            print("Validation", "NDCG@{}:".format(self.args.k), ndcg, "Recall@{}:".format(self.args.k), recall)

            if ndcg > best_metrics:
                best_metrics = ndcg
                state_dict = self.state_dict()
                torch.save(state_dict, os.path.join(self.args.save_path,
                                                    '{}_{}_seed{}_best_model_lr{}_fn{}_block{}_neg{}.pth'.format(
                                                        self.args.task_name, self.args.model_name, self.args.seed, self.args.lr,
                                                        self.args.factor_num, self.args.block_num, self.args.sample_method)))

            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss


class AERecommender(GeneralRecommender):
    def __init__(self, args):
        super(AERecommender, self).__init__(args)
        self.user_num = None
        self.item_num = None
        self.history_user_id, self.history_item_id = None, None
        self.history_user_value, self.history_item_value = None, None

    def get_user_rating_matrix(self, user):
        '''
        just convert the raw rating matrix to a much smaller matrix for calculation,
        the row index will be the new id for uid, but col index will still remain the old iid
        '''
        col_indices = self.history_item_id[user].flatten()  # batch * max_inter_by_user -> (batch * max_inter_by_user)
        row_indices = torch.arange(user.shape[0]).to(self.device).repeat_interleave(
            self.history_item_id.shape[1], dim=0)  # batch -> (batch * max_inter_by_user)
        rating_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.item_num)  # batch * item_num
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())

        return rating_matrix

    def get_item_rating_matrix(self, item):
        col_indices = self.history_user_id[item].flatten()
        row_indices = torch.arange(item.shape[0]).to(self.device).repeat_interleave(
            self.history_user_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1).to(self.device).repeat(item.shape[0], self.user_num)
        rating_matrix.index_put_((row_indices, col_indices), self.history_user_value[item].flatten())

        return rating_matrix

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -(self.gamma + torch.sigmoid(pos_score - neg_score)).log().sum()

        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = torch.clamp(1 - (pos_score - neg_score), min=0).sum()

        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = (neg_score - pos_score).sigmoid().sum() + neg_score.pow(2).sigmoid().sum()

        return loss