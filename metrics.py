# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:13 2021

@author: lpott
"""
import torch
import math
import numpy as np
from tqdm import tqdm


class Recall(object):
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