import torch
import random
import time
import argparse
from utils import *
from trainer import *
from neg_sampler import *
from load_model import *
from splitter import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from model.sequence_model.bert4rec import BERTModel
from model.sequence_model.sasrec import SASRec
from model.sequence_model.nextitnet import NextItNet
from model.sequence_model.gru4rec import GRU4Rec
from model.ctr.deepfm import DeepFM
from model.ctr.xdeepfm import xDeepFM
from model.ctr.nfm import NFM
from model.ctr.wdl import WDL
from model.ctr.afm import AFM
from model.ctr.dcn import DCN
from model.ctr.dcnmix import DCNMix
from model.ctr.dien import DIEN
from model.ctr.din import DIN
from model.transfer_learning.peterrec import PeterRec
from model.mtl.esmm import ESMM
from model.mtl.mmoe import MMOE
from model.model_accelerate.stackrec import StackRec
from model.model_compression.cprec import CpRec
from model.inference_acceleration.skiprec import SkipRec, PolicyNetGumbel
from model.inference_acceleration.sas4infacc import SAS4infaccModel, SAS_PolicyNetGumbel
from model.transfer_learning.sas4transfer import SAS_TransferModel
from model.user_profile_representation.bert4profile import BERT_ProfileModel
from model.user_profile_representation.peter4profile import Peter_ProfileModel
from model.user_profile_representation.dnn4profile import DNNModel
from model.life_long.conure import Conure
from model.life_long.bert4life import BERT4Life
from model.life_long.sas4life import SAS4Life
from model.coldstart.bert4coldstart import BERT_ColdstartModel
from model.coldstart.peter4coldstart import Peter4Coldstart
from model.model_accelerate.sas4acc import SAS4accModel
from model.model_compression.sas4cp import SAS4cpModel
from model.cf.ncf import NCF
from model.cf.mf import MF
from model.cf.lightgcn import LightGCN
from model.cf.ngcf import NGCF
# from model.cf.vae import VAECF
# from model.cf.item2vec import Item2Vec

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def select_sampler(train_data, val_data, test_data, user_count, item_count, args):
    if args.sample == 'random':
        return RandomNegativeSampler(train_data, val_data, test_data, user_count, item_count, args.negsample_size, args.seed, args.negsample_savefolder)
    elif args.sample == 'popular':
        return PopularNegativeSampler(train_data, val_data, test_data, user_count, item_count, args.negsample_size, args.seed, args.negsample_savefolder)

def get_data(args):
    name = args.task_name
    path = args.dataset_path
    rng = random.Random(args.seed)
    if name == 'ctr':
        if args.model_name == 'din' or args.model_name == 'dien':
            train, test, train_model_input, test_model_input, df_columns, hist_list = ctr_din_dataset(path)
            return train, test, train_model_input, test_model_input, df_columns, hist_list
        else:
            train, test, train_model_input, test_model_input, lf_columns, df_columns = ctrdataset(path)
            #share historical embedding
            # train, test, train_model_input, test_model_input, lf_columns, df_columns = ctr_share_dataset(args, path)
            return train, test, train_model_input, test_model_input, lf_columns, df_columns
    elif name == 'sequence' or name == 'transfer_learning' or name == 'model_acc' or name == 'model_compr' or name == 'inference_acc' or name == 'eval':
        _, data, user_count, item_count = sequencedataset(args.item_min, args, path)
        args.num_users = user_count
        args.num_items = item_count
        train_data, val_data, test_data = train_val_test_split(data)
        train_data_s, val_data_s = {}, {}
        data_len = len(train_data)
        i = 0
        for key, _ in val_data.items():
            train_data_s[key] = train_data[key]
            val_data_s[key] = val_data[key]
            i += 1
            if i == int(data_len / args.valid_rate):
                break
        if 'bert' in args.model_name:
            train_dataset = BertTrainDataset(train_data, args.max_len, args.bert_mask_prob, args.pad_token, args.num_items, rng)#
        else:
            train_dataset = BuildTrainDataset(train_data, args.max_len, args.bert_mask_prob, args.pad_token, args.num_items, rng)
        valid_dataset = Build_full_EvalDataset(train_data_s, val_data_s, args.max_len, args.pad_token, args.num_items)
        seq_data = {}
        for key, value in train_data.items():
            tmp = train_data[key] + val_data[key]
            seq_data[key] = tmp
        test_dataset = Build_full_EvalDataset(seq_data, test_data, args.max_len, args.pad_token, args.num_items)
        train_dataloader = get_train_loader(train_dataset, args)
        valid_dataloader = get_val_loader(valid_dataset, args)
        if name == 'inference_acc':
            args.test_batch_size = 1
        test_dataloader = get_test_loader(test_dataset, args)
        return train_dataloader, valid_dataloader, test_dataloader
    elif name == 'cf':
        df, user_count, item_count = new_cf(args)
        args.num_users = user_count
        args.num_items = item_count
        t_splitter = TestSplitter(args)
        train_index, test_index = t_splitter.split(df)
        print("split train and test")
        train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()
        v_splitter = ValidationSplitter(args)
        train_index, val_index = v_splitter.split(train_set)
        print("split train and val")
        train, validation = train_set.iloc[train_index, :].copy(), train_set.iloc[val_index, :].copy()
        val_ur = get_ur(validation)
        train_ur = get_ur(train)
        test_ur = get_ur(test_set)
        args.train_ur = train_ur
        args.val_ur = val_ur
        args.test_ur = test_ur
        val_u = sorted(val_ur.keys())
        args.val_u = val_u
        test_u = sorted(test_ur.keys())
        args.test_u = test_u
        if args.model_name in ['vae']:
            hist_item_id, hist_item_value, _ = get_history_matrix(train, args, row='user_id')
            args.history_item_id, args.history_item_value = hist_item_id, hist_item_value
            train_dataset = AEDataset(train)
            train_loader = get_train_loader(train_dataset, args)
        elif args.model_name in ['mf', 'ncf', 'ngcf', 'lightgcn']:
            if args.model_name in ['ngcf', 'lightgcn']:
                args.inter_matrix = get_inter_matrix(train, args)
            sampler = BasicNegtiveSampler(train, args)
            train_samples = sampler.sampling()
            train_dataset = BasicDataset(train_samples)
            train_loader = get_train_loader(train_dataset, args)
        elif args.model_name in ['item2vec']:
            sampler = SkipGramNegativeSampler(train, args)
            train_samples = sampler.sampling()
            train_dataset = BasicDataset(train_samples)
            train_loader = get_train_loader(train_dataset, args)

        val_dataset = Cf_valDataset(val_u)
        val_loader = get_val_loader(val_dataset, args)
        test_dataset = Cf_valDataset(test_u)
        test_loader = get_test_loader(test_dataset, args)
        return train_loader, val_loader, test_loader

    elif name == 'cold_start':
        if args.ch:
            print("hot & cold")
            cold_data, hot_data, user_count, vocab_size, item_count = colddataset(args.item_min, args)
            size1 = len(cold_data) // 2
            cold_1 = cold_data[:size1]
            cold_2 = cold_data[size1:]
            val_len = len(cold_2) // 2
            train_data = hot_data.append(cold_1)
            val_data = cold_2[:val_len]
            test_data = cold_2[val_len:]
            x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
            x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
            x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()
        else:
            data, user_count, vocab_size, item_count = colddataset(args.item_min, args)
            x_train, x_test, y_train, y_test = train_test_split(data.source.values.tolist(),
                                                                data.target.values.tolist(),
                                                                test_size=0.2, random_state=args.seed)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=args.seed)
        args.num_users = user_count
        args.num_items = item_count
        args.num_embedding = vocab_size

        train_dataset, valid_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token), ColdEvalDataset(
            x_val, y_val, args.max_len, args.pad_token, args.num_items)
        test_dataset = ColdEvalDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items)
        train_dataloader = get_train_loader(train_dataset, args)
        valid_dataloader = get_val_loader(valid_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)
        return train_dataloader, valid_dataloader, test_dataloader
    elif name == 'life_long':
        if args.task == 0:
            _, data, user_count, item_count = sequencedataset(args.item_min, args, path)
            vocab_size = item_count
        else:
            data, user_count, vocab_size, item_count = lifelongdataset(args.item_min, args, path)
        args.num_users = user_count
        args.num_items = item_count
        args.num_embedding = vocab_size
        # args.pad_token = item_count + 1

        if args.task == 0:
            train_data, val_data, test_data = train_val_test_split(data) #
            train_data_s, val_data_s = {}, {}
            data_len = len(train_data)
            i = 0
            for key, _ in val_data.items():
                train_data_s[key] = train_data[key]
                val_data_s[key] = val_data[key]
                i += 1
                if i == int(data_len / 100):
                    break
            if 'bert' in args.model_name:
                train_dataset = BertTrainDataset(train_data, args.max_len, args.bert_mask_prob, args.pad_token, args.num_items, rng)#
            else:
                train_dataset = BuildTrainDataset(train_data, args.max_len, args.bert_mask_prob, args.pad_token, args.num_items, rng)#
            valid_dataset = Build_full_EvalDataset(train_data_s, val_data_s, args.max_len, args.pad_token, args.num_items)
            test_dataset = Build_full_EvalDataset(train_data, test_data, args.max_len, args.pad_token, args.num_items)
        else:
            x_train, x_test, y_train, y_test = train_test_split(data.source.values.tolist(), data.target.values.tolist(),
                                                                test_size=0.2, random_state=args.seed)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=args.seed)
            train_dataset, valid_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token), ColdEvalDataset(
                x_val, y_val, args.max_len, args.pad_token, item_count)
            test_dataset = ColdEvalDataset(x_test, y_test, args.max_len, args.pad_token, item_count)
        train_dataloader = get_train_loader(train_dataset, args)
        valid_dataloader = get_val_loader(valid_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)
        return train_dataloader, valid_dataloader, test_dataloader, vocab_size, item_count
    elif name == 'user_profile_represent':
        df, user_count, item_count, label_count = profiledata(args.item_min, args, path)
        args.num_users = user_count
        args.num_items = item_count
        args.num_labels = label_count
        x_train, x_test, y_train, y_test = train_test_split(df.history.values.tolist(), df.profile.values.tolist(), test_size=0.2, random_state=args.seed)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=args.seed)
        train_dataset, valid_dataset = ProfileDataset(x_train, y_train, args.max_len, args.pad_token), ProfileDataset(x_val, y_val, args.max_len, args.pad_token)
        test_dataset = ProfileDataset(x_test, y_test, args.max_len, args.pad_token)

        train_dataloader = get_train_loader(train_dataset, args)
        valid_dataloader = get_val_loader(valid_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)
        return train_dataloader, valid_dataloader, test_dataloader
    elif name == 'mtl':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = mtl_data(path, args)
        if args.mtl_task_num == 2:
            train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
        else:
            train_dataset = (train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values)
        train_dataset = mtlDataSet(train_dataset, args)
        val_dataset = mtlDataSet(val_dataset, args)
        test_dataset = mtlDataSet(test_dataset, args)

        # dataloader
        train_dataloader = get_train_loader(train_dataset, args)
        val_dataloader = get_val_loader(val_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)

        return train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(args, linear_feature_columns=None, dnn_feature_columns=None, history_feature_list=None):
    name = args.model_name
    if name == 'deepfm':
        return DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'nfm':
        return NFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'xdeepfm':
        return xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'wdl':
        return WDL(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'afm':
        return AFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'dcn':
        return DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'dcnmix':
        return DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
    elif name == 'din':
        return DIN(dnn_feature_columns, history_feature_list, task='binary', device=args.device)
    elif name == 'dien':
        return DIEN(dnn_feature_columns, history_feature_list, task='binary', device=args.device)
    elif name == 'bert4rec':
        return BERTModel(args)
    elif name == 'sasrec':
        return SASRec(args)
    elif name == 'nextitnet':
        return NextItNet(args)
    elif name == 'gru4rec':
        return GRU4Rec(args)
    elif name == 'peterrec':
        return PeterRec(args)
    elif name == 'stackrec':
        return StackRec(args)
    elif name == 'cprec':
        return CpRec(args)
    elif name == 'skiprec':
        return SkipRec(args), PolicyNetGumbel(args)
    elif name == 'sas4infacc':
        return SAS4infaccModel(args), SAS_PolicyNetGumbel(args)
    elif name == 'sas4transfer':
        return SAS_TransferModel(args)
    elif name == 'bert4profile':
        return BERT_ProfileModel(args)
    elif name == 'peter4profile':
        return Peter_ProfileModel(args)
    elif name == 'conure':
        return Conure(args)
    elif name == 'bert4life':
        return BERT4Life(args)
    elif name == 'sas4life':
        return SAS4Life
    elif name == 'bert4coldstart':
        return BERT_ColdstartModel(args)
    elif name == 'peter4coldstart':
        return Peter4Coldstart(args)
    elif name == 'dnn4profile':
        return DNNModel(args)
    elif name == 'sas4acc':
        return SAS4accModel(args)
    elif name == 'sas4cp':
        return SAS4cpModel(args)
    elif name == 'ncf':
        return NCF(args)
    elif name == 'mf':
        return MF(args)
    elif name == 'lightgcn':
        return LightGCN(args)
    elif name == 'ngcf':
        return NGCF(args)
    # elif name == 'vae':
    #     return VAECF(args)
    # elif name == 'item2vec':
    #     return Item2Vec(args)
    else:
        raise ValueError('unknown model name: ' + name)

def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', default='')
    parser.add_argument('--task_num', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--target_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--sample', type=str, default='random')
    parser.add_argument('--negsample_savefolder', type=str, default='./data/neg_data/')
    parser.add_argument('--negsample_size', type=int, default=99)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--item_min', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    # parser.add_argument('--save_path', type=str, default='/data/home')
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--valid_rate', type=int, default=100)

    parser.add_argument('--model_name', default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--re_epochs', type=int, default=20)

    parser.add_argument('--lr', type=float, default=0.0005)

    parser.add_argument('--device', default='cuda')  # cuda:0
    parser.add_argument('--is_parallel', type=bool, default=False)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='l2 regularization') #0.008
    parser.add_argument('--decay_step', type=int, default=5, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR')
    parser.add_argument('--num_users', type=int, default=1, help='Number of total users')
    parser.add_argument('--num_items', type=int, default=1, help='Number of total items')
    parser.add_argument('--num_embedding', type=int, default=1, help='Number of total source items')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of total labels')
    parser.add_argument('--k', type=int, default=20, help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)')
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 20], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

    #model param
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden vectors (model)')
    parser.add_argument('--block_num', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of transformer groups')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for multi-attention')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0.3,
                        help='Probability for masking items in the training sequence')
    parser.add_argument('--factor_num', type=int, default=128)
    #Nextitnet
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size for model')
    parser.add_argument('--dilations', type=int, default=[1, 4], help='Number of transformer layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='Number of heads for multi-attention')
    parser.add_argument('--is_mp', type=bool, default=False, help='Number of heads for multi-attention')
    parser.add_argument('--pad_token', type=int, default=0)
    parser.add_argument('--temp', type=int, default=7)

    #SASRec
    parser.add_argument('--l2_emb', default=0.0, type=float)
    #mtl
    parser.add_argument('--mtl_task_num', type=int, default=1, help='0:like, 1:click, 2:two tasks')

    #CF
    parser.add_argument('--test_method', default='ufo', type=str)
    parser.add_argument('--val_method', default='ufo', type=str)
    parser.add_argument('--test_size', default=0.1, type=float)
    parser.add_argument('--val_size', default=0.1111, type=float)
    parser.add_argument('--cand_num', default=100, type=int)
    parser.add_argument('--sample_method', default='high-pop', type=str) #
    # parser.add_argument('--sample_method', default='uniform', type=str)
    parser.add_argument('--sample_ratio', default=0.3, type=float)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--loss_type', default='BPR', type=str)
    parser.add_argument('--init_method', default='default', type=str)
    parser.add_argument('--optimizer', default='default', type=str)
    parser.add_argument('--early_stop', default=True, type=bool)
    parser.add_argument('--reg_1', default=0.0, type=float)
    parser.add_argument('--reg_2', default=0.0, type=float)
    parser.add_argument('--context_window', default=2, type=int)
    parser.add_argument('--rho', default=0.5, type=float)

    #ngcf
    parser.add_argument('--node_dropout', default=0.1,
                        type=float,
                        help='NGCF: Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', default=0.1,
                        type=float,
                        help='NGCF: Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--hidden_size_list', default=[128, 128], type=list)

    #vae
    parser.add_argument('--latent_dim',
                        type=int,
                        default=128,
                        help='bottleneck layer size for autoencoder')
    parser.add_argument('--anneal_cap',
                        type=float,
                        default=0.2,
                        help='Anneal penalty for VAE KL loss')
    parser.add_argument('--total_anneal_steps',
                        type=int,
                        default=1000)
    #model_KD
    parser.add_argument('--kd', type=bool, default=False, help='True: Knowledge distilling, False: Cprec')
    parser.add_argument('--alpha', default=0.4, type=float)

    #model_acc
    parser.add_argument('--add_num_times', type=int, default=2)

    #transfer learning
    parser.add_argument('--is_pretrain', type=int, default=1, help='0: mean transfer, 1: mean pretrain, 2:mean train full model without transfer')

    #user_profile_represent
    parser.add_argument('--user_profile', type=str, default='gender', help='user_profile: gender, age')

    # life_long
    parser.add_argument('--prun_rate', type=float, default=0)
    parser.add_argument('--ll_max_itemnum', type=int, default=0)
    parser.add_argument('--lifelong_eval', type=bool, default=True)
    parser.add_argument('--task1_out', type=int, default=0)
    parser.add_argument('--task2_out', type=int, default=0)
    parser.add_argument('--task3_out', type=int, default=0)
    parser.add_argument('--task4_out', type=int, default=0)
    parser.add_argument('--eval', type=bool, default=True)

    # cold_start
    parser.add_argument('--ch', type=bool, default=True)

    args = parser.parse_args()
    if args.is_parallel:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device)
    # if 'bert' in args.model_name:
    set_seed(args.seed)
    writer = SummaryWriter()
    print(args)
    if args.task_name == 'ctr':
        if args.model_name == 'din' or args.model_name == 'dien':
            train, test, train_model_input, test_model_input, df_columns, hist_list = get_data(args)
            model = get_model(args, linear_feature_columns=None, dnn_feature_columns=df_columns, history_feature_list=hist_list)
        else:
            train, test, train_model_input, test_model_input, lf_columns, df_columns = get_data(args)
            model = get_model(args, linear_feature_columns=lf_columns, dnn_feature_columns=df_columns, history_feature_list=None)
        # model = get_model(args, lf_columns, df_columns)
        model.compile(args, "adam", "binary_crossentropy",
                      metrics=["auc", "acc"])
        history, best_model = model.fit(train_model_input, train['click'].values, batch_size=args.train_batch_size, epochs=args.epochs, verbose=2,
                            validation_split=0.1111)

        pred_ans = best_model.predict(test_model_input, args.test_batch_size)
        print("test LogLoss", round(log_loss(test['click'].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test['click'].values, pred_ans), 4))
    elif args.task_name == 'sequence':
        print('=============sequence=============')
        train_loader, val_loader, test_loader = get_data(args)
        model = get_model(args)
        SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
        if args.eval:
            best_weight = torch.load(os.path.join(args.save_path,
                                                  '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
            model.load_state_dict(best_weight)
            model = model.to(args.device)
            metrics = Sequence_full_Validate(0, model, test_loader, writer, args, test=False)
            # print('inference_time:', model.all_time)
        writer.close()
    elif args.task_name == 'mtl':

        train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data(args)
        if args.mtl_task_num == 2:
            num_task = 2
        else:
            num_task = 1
        if args.model_name == 'esmm':
            model = ESMM(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, num_task=num_task)
        else:
            model = MMOE(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
        mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args, train=False)
    elif args.task_name == 'transfer_learning':
        print('=============transfer_learning=============')
        train_loader, val_loader, test_loader = get_data(args) #, user_noclick
        if args.is_pretrain == 1:
            print("pretrain")
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args) #, user_noclick
            writer.close()
        elif args.is_pretrain == 2:
            args.is_mp = False
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        else:
            print("transfer")
            best_weight = torch.load(args.pretrain_path)
            if 'peter' in args.model_name:
                args.is_mp = True
                best_weight.pop('item_embedding.weight')
                best_weight.pop('final_layer.weight')
                best_weight.pop('final_layer.bias')
            elif 'bert' in args.model_name:
                best_weight.pop('bert.embedding.token.weight')
                best_weight.pop('bert.embedding.position.pe.weight')
                best_weight.pop('out.weight')
                best_weight.pop('out.bias')
            model = get_model(args)
            model_state = model.module.state_dict() if args.is_parallel else model.state_dict()
            best_weight = {k: v for k, v in best_weight.items() if k in model_state}
            model_state.update(best_weight)
            model.load_state_dict(model_state)
            if 'peter' in args.model_name:
                for name, parm in model.named_parameters():
                    if 'item_embedding' not in name and 'mp' not in name and 'final_layer' not in name:
                        parm.requires_grad = False
            else:
                for name, parm in model.named_parameters():
                    if 'embedding' not in name and 'out' not in name:
                        parm.requires_grad = False
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        if args.eval:
            model = get_model(args)
            best_weight = torch.load(os.path.join(args.save_path,
                                                  '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
            model.load_state_dict(best_weight)
            model = model.to(args.device)
            metrics = Sequence_full_Validate(0, model, test_loader, writer, args)

    elif args.task_name == 'model_acc':
        train_loader, val_loader, test_loader = get_data(args)
        if args.is_pretrain == 1:
            print('++++++++++pretrain++++++++++++')
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        elif args.is_pretrain == 2:
            args.block_num = args.block_num * args.add_num_times
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        else:
            print('++++++++++transfer++++++++++')
            best_weight = torch.load(args.pretrain_path)
            last_model = get_model(args)
            last_model.load_state_dict(best_weight)
            last_block_num = args.block_num
            args.block_num = args.block_num * args.add_num_times
            print('block_num', args.block_num)
            model = get_model(args)
            model = new_adj_stack(model, last_model, last_block_num, args)

            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        if args.eval:
            # args.block_num = args.block_num * 2
            model = get_model(args)
            best_weight = torch.load(os.path.join(args.save_path,
                                                  '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
            model.load_state_dict(best_weight)
            model = model.to(args.device)
            metrics = Sequence_full_Validate(0, model, test_loader, writer, args)

    elif args.task_name == 'model_compr':
        train_loader, val_loader, test_loader = get_data(args)  # , user_noclick


        if args.kd:
            student_model = get_model(args)

            tmp_eb = args.embedding_size
            tmp_block = args.block_num
            tmp_hd = args.hidden_size

            args.embedding_size = tmp_eb * 2
            args.block_num = tmp_block * 2
            args.hidden_size = tmp_hd * 2

            teacher_model = get_model(args)

            args.embedding_size = tmp_eb
            args.block_num = tmp_block
            args.hidden_size = tmp_hd

            best_weight = torch.load(args.pretrain_path)
            teacher_model.load_state_dict(best_weight)
            KDTrain(args.epochs, teacher_model, student_model, train_loader, val_loader, writer, args)
        else:
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)  # , user_noclick
        writer.close()

        if args.eval:
            best_weight = torch.load(os.path.join(args.save_path, '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
            model.load_state_dict(best_weight)
            model = model.to(args.device)
            metrics = Sequence_full_Validate(0, model, test_loader, writer, args)
    elif args.task_name == 'inference_acc':
        train_loader, val_loader, test_loader = get_data(args)
        backbonenet, policynet = get_model(args)
        Infacc_Train(args.epochs, backbonenet, policynet, train_loader, val_loader, writer, args)

        if args.eval:
            best_weight1 = torch.load(os.path.join(args.save_path, '{}_{}_seed{}_lr{}_block{}_best_policynet.pth'.format(args.task_name, args.model_name, args.seed,
                                                                                                                                       args.lr, args.block_num)))
            best_weight2 = torch.load(os.path.join(args.save_path,
                                                  '{}_{}_seed{}_lr{}_block{}_best_backbone.pth'.format(
                                                      args.task_name, args.model_name, args.seed, args.lr, args.block_num)))
            policynet.load_state_dict(best_weight1)
            backbonenet.load_state_dict(best_weight2)
            policynet = policynet.to(args.device)
            backbonenet = backbonenet.to(args.device)
            metrics = Infacc_Validate(0, backbonenet, policynet, test_loader, writer, args, test=True)
            print('inference_time:', backbonenet.all_time)
        writer.close()
    elif args.task_name == 'user_profile_represent':
        print('=============user_profile_represent=============')
        train_loader, val_loader, test_loader = get_data(args)
        if args.is_pretrain == 0:
            print('transfer')
            best_weight = torch.load(args.pretrain_path)

            if 'peter' in args.model_name:
                args.is_mp = True
                best_weight.pop('final_layer.weight')
                best_weight.pop('final_layer.bias')
            elif 'bert' in args.model_name:
                best_weight.pop('out.weight')
                best_weight.pop('out.bias')
            model = get_model(args)
            model_state = model.module.state_dict() if args.is_parallel else model.state_dict()
            best_weight = {k: v for k, v in best_weight.items() if k in model_state}
            model_state.update(best_weight)
            model.load_state_dict(model_state)
            if 'peter' in args.model_name:
                for name, parm in model.named_parameters():
                    if 'item_embedding' not in name and 'mp' not in name and 'final_layer' not in name:
                        parm.requires_grad = False
            ProfileTrain(args.epochs, model, train_loader, val_loader, args)
        else:
            model = get_model(args)
            ProfileTrain(args.epochs, model, train_loader, val_loader, args)
        if args.eval:
            if args.is_pretrain == 0:
                args.is_mp = True
            model = get_model(args)
            best_weight = torch.load(os.path.join(args.save_path,
                                                '{}_{}_seed{}_profile-{}_pretrain{}_best_model.pth'.format(args.task_name,
                                                                                     args.model_name, args.seed, args.user_profile, args.is_pretrain)))
            model.load_state_dict(best_weight)
            model = model.to(args.device)
            acc = ProfileValidate(0, model, test_loader, args)

    elif args.task_name == 'life_long':
        train_loader_list = []
        val_loader_list = []
        test_loader_list = []
        retrain_task1 = True
        for i in range(0, args.task_num):
            args.task = i
            args.source_path = '/data/task_0_new.csv'
            if i > 0:
                args.target_path = '/data/task_{}_new.csv'.format(i)
            else:
                args.dataset_path = '/data/task_0_new.csv'
            train_loader1, val_loader1, test_loader1, source_num, target_num = get_data(args)
            args.task1_out = source_num
            if i == 1:
                args.task2_out = target_num
            elif i == 2:
                args.task3_out = target_num
            elif i == 3:
                args.task4_out = target_num
            train_loader_list.append(train_loader1)
            val_loader_list.append(val_loader1)
            test_loader_list.append(test_loader1)
        j = 0
        for i in range(0, args.task_num):
            args.task = i
            if i == 0:
                args.prun_rate = 0.6
            elif i == 1:
                args.prun_rate = 0.3333
                args.lr = 0.0001
                args.train_batch_size = 64
                print(args.lr)
            elif i == 2:
                args.epochs = 10
                args.re_epochs = 10
                args.prun_rate = 0.25
                args.lr = 0.001
                print(args.lr)
            else:
                args.lr = 0.0005
                print(args.lr)
            print("+++++++++task_{}+++++++++".format(i))
            train_loader, val_loader, test_loader = train_loader_list[j], val_loader_list[j], test_loader_list[j]
            j += 1

            model = get_model(args)
            compare_base = False
            if compare_base:
                best_model = SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
                metrics = Sequence_full_Validate(0, best_model, test_loader, writer, args)
                args.lifelong_eval = False
            else:
                if i != 0:
                    for name, parm in model.named_parameters():
                        if 'embedding' in name:
                            parm.requires_grad = False
                            break
                lifelong_Train(args.epochs, model, train_loader, val_loader, writer, i, args)
                if i != args.task_num - 1:
                    print('lr:', args.lr)
                    lifelong_ReTrain(args.re_epochs, model, train_loader, val_loader, test_loader, writer, i, args)

        if args.lifelong_eval:
            i = 0

            model_path2 = os.path.join(args.save_path,
                                       '{}_{}_seed{}_task_3_best_model.pth'.format(args.task_name, args.model_name,
                                                                                   args.seed))
            best_weight2 = torch.load(model_path2, map_location=torch.device(args.device))
            for test_loader in test_loader_list:

                print("++++++++++task{}_test++++++++++".format(i))
                args.task = i
                model = get_model(args)
                model = model.to(args.device)

                model.load_state_dict(best_weight2)
                compare_base = 0
                if compare_base:
                    metrics = Sequence_full_Validate(0, model, test_loader, writer, args)
                    i += 1
                else:
                    if i == 0:
                        current_mask = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, i)),
                                                  map_location=torch.device('cuda'))
                    elif i == 1:
                        current_mask1 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, i - 1)),
                                                   map_location=torch.device('cuda'))
                        current_mask2 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, i)),
                                                   map_location=torch.device('cuda'))
                        current_mask = concat_mask(current_mask1, current_mask2)
                    elif i == 2:
                        current_mask0 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, i - 2)),
                                                   map_location=torch.device('cuda'))
                        current_mask1 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, i - 1)),
                                                   map_location=torch.device('cuda'))
                        current_mask2 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, i)),
                                                   map_location=torch.device('cuda'))
                        # current_mask2 = reverse_mask(current_mask2)
                        current_mask = concat_mask(current_mask0, current_mask1)
                        current_mask = concat_mask(current_mask, current_mask2)
                    if i != 3:
                        model_mask(model, current_mask, args)
                    metrics = Sequence_full_Validate(0, model, test_loader, writer, args)
                    i += 1

    elif args.task_name == 'cold_start':
        print('=============cold_start=============')
        # args.source_path = '/data/sbr_data_1M.csv'
        # args.target_path = '/data/cold_data.csv'
        train_loader, val_loader, test_loader = get_data(args) #, user_noclick
        if args.is_pretrain == 1:
            print("pretrain")
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args) #, user_noclick
            writer.close()
        elif args.is_pretrain == 2:
            args.is_mp = False
            model = get_model(args)
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        else:
            print("transfer")
            best_weight = torch.load(args.pretrain_path)
            if 'peter' in args.model_name:
                args.is_mp = True
                best_weight.pop('final_layer.weight')
                best_weight.pop('final_layer.bias')
            if 'bert' in args.model_name:
                best_weight.pop('out.weight')
                best_weight.pop('out.bias')

            model = get_model(args)

            model_state = model.module.state_dict() if args.is_parallel else model.state_dict()
            best_weight = {k: v for k, v in best_weight.items() if k in model_state}
            model_state.update(best_weight)
            model.load_state_dict(model_state)
            # if 'peter' in args.model_name:
            #     for name, parm in model.named_parameters():
            #         if 'rez' not in name and 'mp' not in name and 'final_layer' not in name:
            #             parm.requires_grad = False
            # else:
            #     for name, parm in model.named_parameters():
            #         if 'out' not in name:
            #             parm.requires_grad = False
            SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
            writer.close()
        if args.eval:
            model = get_model(args)
            best_weight = torch.load(os.path.join(args.save_path,
                                                  '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
            model.load_state_dict(best_weight)
            model = model.to(args.device)
            metrics = Sequence_full_Validate(0, model, test_loader, writer, args)
    elif args.task_name == 'cf':
        metrics_config = {
            "recall": Recall,
            "mrr": MRR,
            "ndcg": NDCG,
            "hr": HR,
            "map": MAP,
            "precision": Precision,
        }
        print('=============cf=============')
        train_loader, val_loader, test_loader = get_data(args) #, user_noclick
        model = get_model(args)
        model.fit(train_loader, val_loader)
        #test
        model = get_model(args)
        print("model:", args.model_name, "neg_sample_method:", args.sample_method)
        best_weight = torch.load(os.path.join(args.save_path,
                                                    '{}_{}_seed{}_best_model_lr{}_fn{}_block{}_neg{}.pth'.format(
                                                        args.task_name, args.model_name, args.seed, args.lr,
                                                        args.factor_num, args.block_num, args.sample_method)))
        model.load_state_dict(best_weight)
        model = model.to(args.device)
        preds = model.rank(test_loader)
        ndcg = NDCG(args.test_ur, preds, args.test_u)
        recall = Recall(args.test_ur, preds, args.test_u)
        print("Test", "NDCG@{}:".format(args.k), ndcg, "Recall@{}:".format(args.k), recall)





