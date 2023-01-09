'''
Reference:
	[1]Xiangnan He et al. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web, pages 173–182, 2017.
Reference:
    https://github.com/recsys-benchmark/DaisyRec-v2.0
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
from model.cf.AbstractRecommender import GeneralRecommender

class NCF(GeneralRecommender):
	def __init__(self, args):
		super(NCF, self).__init__(args)
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""

		self.epochs = args.epochs
		self.lr = args.lr
		self.reg_1 = args.reg_1
		self.reg_2 = args.reg_2
		self.topk = args.k
		self.user_num = args.num_users
		self.item_num = args.num_items
		self.dropout = args.dropout
		self.factor_num = args.factor_num
		self.num_layers = args.block_num
		# self.model = model
		# self.GMF_model = GMF_model
		# self.MLP_model = MLP_model

		self.embed_user_GMF = nn.Embedding(self.user_num, self.factor_num)
		self.embed_item_GMF = nn.Embedding(self.item_num, self.factor_num)
		self.embed_user_MLP = nn.Embedding(
				self.user_num, self.factor_num * (2 ** (self.num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
				self.item_num, self.factor_num * (2 ** (self.num_layers - 1)))

		MLP_modules = []
		for i in range(self.num_layers):
			input_size = self.factor_num * (2 ** (self.num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		# if self.model in ['MLP', 'GMF']:
		# 	predict_size = factor_num
		# else:
		predict_size = self.factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)

		self.loss_type = args.loss_type
		self.optimizer = args.optimizer if args.optimizer != 'default' else 'adam'
		self.initializer = args.init_method if args.init_method != 'default' else 'xavier_normal'
		self.early_stop = args.early_stop
		self.apply(self._init_weight)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.predict_layer.weight,
								a=1, nonlinearity='sigmoid')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()

	def forward(self, user, item):
		embed_user_GMF = self.embed_user_GMF(user)
		embed_item_GMF = self.embed_item_GMF(item)
		output_GMF = embed_user_GMF * embed_item_GMF
		embed_user_MLP = self.embed_user_MLP(user)
		embed_item_MLP = self.embed_item_MLP(item)
		interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
		output_MLP = self.MLP_layers(interaction)

		concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)

	def calc_loss(self, batch):
		user = batch[0].to(self.device).long()
		pos_item = batch[1].to(self.device).long()
		pos_pred = self.forward(user, pos_item)

		if self.loss_type.upper() in ['CL', 'SL']:
			label = batch[2].to(self.device).float()
			loss = self.criterion(pos_pred, label)

			loss += self.reg_1 * (self.embed_item_GMF(pos_item).norm(p=1))
			loss += self.reg_1 * (self.embed_item_MLP(pos_item).norm(p=1))
			loss += self.reg_2 * (self.embed_item_GMF(pos_item).norm())
			loss += self.reg_2 * (self.embed_item_MLP(pos_item).norm())
		elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
			neg_item = batch[2].to(self.device).long()
			neg_pred = self.forward(user, neg_item)
			loss = self.criterion(pos_pred, neg_pred)

			loss += self.reg_1 * (self.embed_item_GMF(pos_item).norm(p=1) + self.embed_item_GMF(neg_item).norm(p=1))
			loss += self.reg_1 * (self.embed_item_MLP(pos_item).norm(p=1) + self.embed_item_GMF(neg_item).norm(p=1))
			loss += self.reg_2 * (self.embed_item_GMF(pos_item).norm() + self.embed_item_GMF(neg_item).norm())
			loss += self.reg_2 * (self.embed_item_MLP(pos_item).norm() + self.embed_item_GMF(neg_item).norm())
		else:
			raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

		loss += self.reg_1 * (self.embed_user_GMF(user).norm(p=1))
		loss += self.reg_1 * (self.embed_user_MLP(user).norm(p=1))
		loss += self.reg_2 * (self.embed_user_GMF(user).norm())
		loss += self.reg_2 * (self.embed_user_MLP(user).norm())

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
				us = us.to(self.device)
				rank_list = self.full_rank(us)
				rec_ids = torch.cat((rec_ids, rank_list), 0)

		return rec_ids.cpu().numpy()

	def full_rank(self, u):
		embed_user_GMF = self.embed_user_GMF(u).unsqueeze(dim=1)  # factor
		embed_item_GMF = self.embed_item_GMF.weight.unsqueeze(0).repeat(embed_user_GMF.shape[0], 1, 1)  # item * factor
		output_GMF = embed_user_GMF * embed_item_GMF  # item * factor
		embed_user_MLP = self.embed_user_MLP(u).unsqueeze(dim=1)  # factor
		embed_item_MLP = self.embed_item_MLP.weight.unsqueeze(0).repeat(embed_user_GMF.shape[0], 1, 1)  # item * factor
		interaction = torch.cat((embed_user_MLP.expand_as(embed_item_MLP), embed_item_MLP),
								dim=-1)  # item * (2*factor)
		output_MLP = self.MLP_layers(interaction)  # item * dim

		concat = torch.cat((output_GMF, output_MLP), -1)  # item * (dim + factor)
		scores = self.predict_layer(concat).squeeze()  # item

		return torch.argsort(scores, descending=True)[:, :self.topk]
