# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import os
import pandas as pd
from tqdm import tqdm



class recall(object):
    def __init__(self, user_noclick, n_users, n_items, k=10):
        print("=" * 10, "Creating Hit@{:d} Metric Object".format(k), "=" * 10)

        self.user_noclick = user_noclick
        self.n_users = n_users
        self.n_items = n_items
        self.k = k

    def __call__(self, model, dataloader):

        model.eval()
        with torch.no_grad():

            total_hits = 0
            for data in tqdm(dataloader):
                inputs, labels, x_lens, uid = data
                outputs = model(inputs.cuda())

                for i, uid in enumerate(uid.squeeze()):
                    negatives, probabilities = self.user_noclick[uid.item()]
                    sampled_negatives = np.random.choice(negatives, size=100, replace=False,
                                                         p=probabilities).tolist() + [
                                            labels[i, x_lens[i].item() - 1].item()]

                    topk_items = outputs[i, x_lens[i].item() - 1, sampled_negatives].argsort(0, descending=True)[
                                 :self.k]
                    total_hits += torch.sum(topk_items == 100).cpu().item()

        return total_hits / self.n_users * 100

def sample_top_k(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    # probs = a[idx]
    # probs = probs / np.sum(probs)
    # choice = np.random.choice(idx, p=probs)
    return idx

def ndcg_accuracy(output, target, curr_preds_5, rec_preds_5, ndcg_preds_5, curr_preds_20 ,rec_preds_20, ndcg_preds_20, topk): # output: [batch_size, item_size] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # global curr_preds_5
    # global rec_preds_5
    # global ndcg_preds_5
    # global curr_preds_20
    # global rec_preds_20
    # global ndcg_preds_20


    for bi in range(output.shape[0]):
        pred_items_5 = sample_top_k(output[bi], top_k=topk[0])  # top_k=5
        pred_items_20 = sample_top_k(output[bi], top_k=topk[1])

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = pred_items_20.get(true_item)
        if rank_5 == None:
            curr_preds_5.append(0.0)
            rec_preds_5.append(0.0)
            ndcg_preds_5.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5.append(MRR_5)
            rec_preds_5.append(Rec_5)#4
            ndcg_preds_5.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20.append(0.0)
            rec_preds_20.append(0.0)#2
            ndcg_preds_20.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20.append(MRR_20)
            rec_preds_20.append(Rec_20)#4
            ndcg_preds_20.append(ndcg_20)  # 4
    metrics = {'mrr_5': sum(curr_preds_5) / float(len(curr_preds_5)),
               'mrr_20': sum(curr_preds_20) / float(len(curr_preds_20)),
               'hit_5': sum(rec_preds_5) / float(len(rec_preds_5)),
               'hit_20': sum(rec_preds_20) / float(len(rec_preds_20)),
               'ndcg_5': sum(ndcg_preds_5) / float(len(ndcg_preds_5)),
               'ndcg_20': sum(ndcg_preds_20) / float(len(ndcg_preds_20))}
    # if batch_idx % max(10, batch_num//10) == 0:
    #     print("epoch/total_epoch: {}/{}\t batch/total_batches: {}/{} \t loss: {:.3f}".format(
    #                 epoch, args.epochs, batch_idx,  batch_num, loss/(batch_idx+1)))
    #     print("epoch/total_epoch: {}/{}\t batch/total_batches: {}/{}".format(
    #         epoch, args.epochs, batch_idx, batch_num))

    # print("Accuracy hit_5: {}".format(sum(rec_preds_5) / float(len(rec_preds_5))))  # 5
    # print("Accuracy hit_20: {}".format(sum(rec_preds_20) / float(len(rec_preds_20))))  # 5
    return metrics

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def cf_metrics(model, test_loader, top_k, device, args):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)
        if args.model_name == 'vae':
            rating_matrix = model.get_user_rating_matrix(user)
            predictions, _, _ = model.forward(rating_matrix)
            predictions = predictions.sum(-1).view(-1)
        else:
            predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
                item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


metrics_name_config = {
    "recall": 'Recall',
    "mrr": 'MRR',
    "ndcg": 'NDCG',
    "hit": 'Hit Ratio',
    "precision": 'Precision',
    "f1": 'F1-score',
    "auc": 'AUC',
    "coverage": 'Coverage',
    "diversity": 'Diversity',
    "popularity": 'Average Popularity',
}


def calc_ranking_results(test_ur, pred_ur, test_u, config):
    '''
    calculate metrics with prediction results and candidates sets

    Parameters
    ----------
    test_ur : defaultdict(set)
        groud truths for user in test set
    pred_ur : np.array
        rank list for user in test set
    test_u : list
        the user in order from test set
    '''
    logger = config['logger']
    path = config['res_path']
    if not os.path.exists(path):
        os.makedirs(path)

    metric = Metric(config)
    res = pd.DataFrame({
        'KPI@K': [metrics_name_config[kpi_name] for kpi_name in config['metrics']]
    })

    common_ks = [1, 5, 10, 20, 30, 50]
    if config['topk'] not in common_ks:
        common_ks.append(config['topk'])
    for topk in common_ks:
        if topk > config['topk']:
            continue
        else:
            rank_list = pred_ur[:, :topk]
            kpis = metric.run(test_ur, rank_list, test_u)
            if topk == 10:
                for kpi_name, kpi_res in zip(config['metrics'], kpis):
                    kpi_name = metrics_name_config[kpi_name]
                    logger.info(f'{kpi_name}@{topk}: {kpi_res:.4f}')

            res[topk] = np.array(kpis)

    return res


class Metric(object):
    def __init__(self, config) -> None:
        self.metrics = config['metrics']
        self.item_num = config['item_num']
        self.item_pop = config['item_pop'] if 'coverage' in self.metrics else None
        self.i_categories = config['i_categories'] if 'diversity' in self.metrics else None

    def run(self, test_ur, pred_ur, test_u):
        res = []
        for mc in self.metrics:
            if mc == "coverage":
                kpi = Coverage(pred_ur, self.item_num)
            elif mc == "popularity":
                kpi = Popularity(test_ur, pred_ur, test_u, self.item_pop)
            elif mc == "diversity":
                kpi = Diversity(pred_ur, self.i_categories)
            elif mc == 'ndcg':
                kpi = NDCG(test_ur, pred_ur, test_u)
            elif mc == 'mrr':
                kpi = MRR(test_ur, pred_ur, test_u)
            elif mc == 'recall':
                kpi = Recall(test_ur, pred_ur, test_u)
            elif mc == 'precision':
                kpi = Precision(test_ur, pred_ur, test_u)
            elif mc == 'hit':
                kpi = HR(test_ur, pred_ur, test_u)
            elif mc == 'map':
                kpi = MAP(test_ur, pred_ur, test_u)
            elif kpi == 'f1':
                kpi = F1(test_ur, pred_ur, test_u)
            elif kpi == 'auc':
                kpi = AUC(test_ur, pred_ur, test_u)
            else:
                raise ValueError(f'Invalid metric name {mc}')

            res.append(kpi)

        return res


def Coverage(pred_ur, item_num):
    '''
    Ge, Mouzhi, Carla Delgado-Battenfeld, and Dietmar Jannach. "Beyond accuracy: evaluating recommender systems by coverage and serendipity." Proceedings of the fourth ACM conference on Recommender systems. 2010.
    '''
    return len(np.unique(pred_ur)) / item_num


def Popularity(test_ur, pred_ur, test_u, item_pop):
    '''
    Abdollahpouri, Himan, et al. "The unfairness of popularity bias in recommendation." arXiv preprint arXiv:1907.13286 (2019).

    \frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}
    '''
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        i = np.intersect1d(pred, list(gt))
        if len(i):
            avg_pop = np.sum(item_pop[i]) / len(gt)
            res.append(avg_pop)
        else:
            res.append(0)

    return np.mean(res)


def Diversity(pred_ur, i_categories):
    '''
    Intra-list similarity for diversity

    Parameters
    ----------
    pred_ur : np.array
        rank list for each user in test set
    i_categories : np.array
        (item_num, category_num) with 0/1 value
    '''
    res = []
    for u in range(len(pred_ur)):
        ILD = []
        for i in range(len(pred_ur[u])):
            item_i_cats = i_categories[pred_ur[u, i]]
            for j in range(i + 1, len(pred_ur[u])):
                item_j_cats = i_categories[pred_ur[u, j]]
                distance = np.linalg.norm(item_i_cats - item_j_cats)
                ILD.append(distance)
        res.append(np.mean(ILD))

    return np.mean(res)


def Precision(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        pre = np.in1d(pred, list(gt)).sum() / len(pred)

        res.append(pre)

    return np.mean(res)


def Recall(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        rec = np.in1d(pred, list(gt)).sum() / len(gt)

        res.append(rec)

    return np.mean(res)


def MRR(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        mrr = 0.
        for index, item in enumerate(pred):
            if item in gt:
                mrr = 1 / (index + 1)
                break

        res.append(mrr)

    return np.mean(res)


def MAP(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        r = np.in1d(pred, list(gt))
        out = [r[:k + 1].sum() / (k + 1) for k in range(r.size) if r[k]]
        if not out:
            res.append(0.)
        else:
            ap = np.mean(out)
            res.append(ap)

    return np.mean(res)

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 1)+1),
        # np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg



def NDCG(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        nd = getNDCG(pred, gt)
        res.append(nd)
    return np.mean(res)


def HR(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        res.append(1 if r.sum() else 0)

    return np.mean(res)


def AUC(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pos_num = r.sum()
        neg_num = len(pred) - pos_num

        pos_rank_num = 0
        for j in range(len(r) - 1):
            if r[j]:
                pos_rank_num += np.sum(~r[j + 1:])

        auc = pos_rank_num / (pos_num * neg_num)
        res.append(auc)

    return np.mean(res)


def F1(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pre = r.sum() / len(pred)
        rec = r.sum() / len(gt)

        f1 = 2 * pre * rec / (pre + rec)
        res.append(f1)

    return np.mean(res)