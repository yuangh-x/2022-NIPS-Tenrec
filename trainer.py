import os.path
import time
from loggers import *
from copy import deepcopy
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from metrics import *

def mtlTrain(model, train_loader, val_loader, test_loader, args, train=True):
    device = args.device
    epoch = args.epochs
    early_stop = 5
    path = os.path.join(args.save_path, '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed, args.mtl_task_num))
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    # train
    if args.mtl_task_num == 2:
        model.train()
        for i in range(epoch):
            y_train_click_true = []
            y_train_click_predict = []
            y_train_like_true = []
            y_train_like_predict = []
            total_loss, count = 0, 0
            for idx, (x, y1, y2) in enumerate(train_loader):
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                predict = model(x)
                y_train_click_true += list(y1.squeeze().cpu().numpy())
                y_train_like_true += list(y2.squeeze().cpu().numpy())
                y_train_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
                y_train_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
                loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
                loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
                loss = loss_1 + loss_2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                count += 1
            click_auc = roc_auc_score(y_train_click_true, y_train_click_predict)
            like_auc = roc_auc_score(y_train_like_true, y_train_like_predict)
            print("Epoch %d train loss is %.3f, click auc is %.3f and like auc is %.3f" % (i + 1, total_loss / count,
                                                                                             click_auc, like_auc))
            # 验证
            total_eval_loss = 0
            model.eval()
            count_eval = 0
            y_val_click_true = []
            y_val_like_true = []
            y_val_click_predict = []
            y_val_like_predict = []
            for idx, (x, y1, y2) in enumerate(val_loader):
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                predict = model(x)
                y_val_click_true += list(y1.squeeze().cpu().numpy())
                y_val_like_true += list(y2.squeeze().cpu().numpy())
                y_val_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
                y_val_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
                loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
                loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
                loss = loss_1 + loss_2
                total_eval_loss += float(loss)
                count_eval += 1
            click_auc = roc_auc_score(y_val_click_true, y_val_click_predict)
            like_auc = roc_auc_score(y_val_like_true, y_val_like_predict)
            print("Epoch %d val loss is %.3f, click auc is %.3f and like auc is %.3f" % (i + 1,
                                                                                        total_eval_loss / count_eval,
                                                                                        click_auc, like_auc))

            # earl stopping
            if i == 0:
                eval_loss = total_eval_loss / count_eval
            else:
                if total_eval_loss / count_eval < eval_loss:
                    eval_loss = total_eval_loss / count_eval
                    state = model.state_dict()
                    torch.save(state, path)
                else:
                    if patience < early_stop:
                        patience += 1
                    else:
                        print("val loss is not decrease in %d epoch and break training" % patience)
                        break
        #test
        state = torch.load(path)
        model.load_state_dict(state)
        total_test_loss = 0
        model.eval()
        count_eval = 0
        y_test_click_true = []
        y_test_like_true = []
        y_test_click_predict = []
        y_test_like_predict = []
        for idx, (x, y1, y2) in enumerate(test_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_test_click_true += list(y1.squeeze().cpu().numpy())
            y_test_like_true += list(y2.squeeze().cpu().numpy())
            y_test_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_test_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_test_loss += float(loss)
            count_eval += 1
        click_auc = roc_auc_score(y_test_click_true, y_test_click_predict)
        like_auc = roc_auc_score(y_test_like_true, y_test_like_predict)
        print("Epoch %d test loss is %.3f, click auc is %.3f and like auc is %.3f" % (i + 1,
                                                                                     total_test_loss / count_eval,
                                                                                     click_auc, like_auc))

    else:
        if train:
            model.train()
            for i in range(epoch):
                y_train_label_true = []
                y_train_label_predict = []
                total_loss, count = 0, 0
                for idx, (x, y) in enumerate(train_loader):
                    x, y = x.to(device), y.to(device)
                    predict = model(x)
                    y_train_label_true += list(y.squeeze().cpu().numpy())
                    y_train_label_predict += list(predict[0].squeeze().cpu().detach().numpy())
                    loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
                    loss = loss_1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)
                    count += 1
                auc = roc_auc_score(y_train_label_true, y_train_label_predict)
                print("Epoch %d train loss is %.3f, auc is %.3f" % (i + 1, total_loss / count, auc))
                # 验证
                total_eval_loss = 0
                model.eval()
                count_eval = 0
                y_val_label_true = []
                y_val_label_predict = []
                for idx, (x, y) in enumerate(val_loader):
                    x, y = x.to(device), y.to(device)
                    predict = model(x)
                    y_val_label_true += list(y.squeeze().cpu().numpy())
                    y_val_label_predict += list(predict[0].squeeze().cpu().detach().numpy())
                    loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
                    loss = loss_1
                    total_eval_loss += float(loss)
                    count_eval += 1
                auc = roc_auc_score(y_val_label_true, y_val_label_predict)
                print("Epoch %d val loss is %.3f, auc is %.3f " % (i + 1, total_eval_loss / count_eval,
                                                                                             auc))
                # earl stopping
                if i == 0:
                    eval_loss = total_eval_loss / count_eval
                else:
                    if total_eval_loss / count_eval < eval_loss:
                        eval_loss = total_eval_loss / count_eval
                        state = model.state_dict()
                        torch.save(state, path)
                    else:
                        if patience < early_stop:
                            patience += 1
                        else:
                            print("val loss is not decrease in %d epoch and break training" % patience)
                            break

        total_test_loss = 0
        model.eval()
        count_eval = 0
        y_test_label_true = []
        y_test_label_predict = []
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            predict = model(x)
            y_test_label_true += list(y.squeeze().cpu().numpy())
            y_test_label_predict += list(predict[0].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
            loss = loss_1
            total_test_loss += float(loss)
            count_eval += 1
        auc = roc_auc_score(y_test_label_true, y_test_label_predict)
        print("Epoch %d test loss is %.3f, auc is %.3f" % (i + 1, total_test_loss / count_eval,
                                                                                      auc))

def Infacc_Train(epochs, b_model, p_model, train_loader, val_loader, writer, args): #, user_noclicks
    b_optimizer = torch.optim.Adam(b_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    p_optimizer = torch.optim.Adam(p_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_model = b_model.to(args.device)
    p_model = p_model.to(args.device)
    best_metric = 0
    for epoch in range(epochs):
        Infacc_Trainer(epoch, b_model, p_model, train_loader, b_optimizer, p_optimizer, writer, args)
        metrics = Infacc_Validate(epoch, b_model, p_model, val_loader, writer, args)

        current_metric = metrics['NDCG@5']
        if best_metric < current_metric:
            best_metric = current_metric
            p_state_dict = p_model.state_dict()
            b_state_dict = b_model.state_dict()
            torch.save(p_state_dict, os.path.join(args.save_path, '{}_{}_seed{}_lr{}_block{}_best_policynet.pth'.format(args.task_name, args.model_name, args.seed,
                                                                                                                                       args.lr, args.block_num)))
            torch.save(b_state_dict, os.path.join(args.save_path,
                                                  '{}_{}_seed{}_lr{}_block{}_best_backbone.pth'.format(
                                                      args.task_name, args.model_name, args.seed, args.lr, args.block_num)))

def Infacc_Trainer(epoch, b_model, p_model, dataloader, b_optimizer, p_optimizer, writer, args):
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    b_model.train()
    p_model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    # for data in tqdm(dataloader):
    for data in dataloader:
        b_optimizer.zero_grad()
        p_optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data
        policy_action = p_model(seqs)
        logits = b_model(seqs, policy_action)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = loss_fn(logits, labels)
        loss.backward()
        p_optimizer.step()
        b_optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))

def Infacc_pn_Trainer(epoch, b_model, p_model, dataloader, b_optimizer, p_optimizer, writer, args):
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    b_model.train()
    p_model.train()
    running_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    for data in tqdm(dataloader):
        b_optimizer.zero_grad()
        p_optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, pos, neg = data
        policy_action = p_model(seqs)

        pos_logits, neg_logits = b_model(seqs, policy_action, pos, neg)
        pos_labels = torch.ones(pos_logits.shape, device=args.device)
        neg_labels = torch.zeros(neg_logits.shape, device=args.device)

        indices = torch.where(pos != 0)
        loss = loss_fn(pos_logits[indices], pos_labels[indices])
        loss += loss_fn(neg_logits[indices], neg_labels[indices])
        loss.backward()
        p_optimizer.step()
        b_optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))

def Infacc_Validate(epoch, b_model, p_model, dataloader, writer, args, test=False):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    p_model.eval()
    b_model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        tqdm_dataloader = dataloader
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            policy_action = p_model(seqs)
            if test:
                scores = b_model.predict(seqs, policy_action)
            else:
                scores = b_model(seqs, policy_action)
            scores = scores[:, -1, :]  # B x V
            metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
        for key, value in avg_metrics.items():
            avg_metrics[key] = value / i
        print(avg_metrics)
        for k in sorted(args.metric_ks, reverse=True):
            writer.add_scalar('Train/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
        return avg_metrics

def Infacc_pn_Validate(epoch, b_model, p_model, dataloader, writer, args):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    p_model.eval()
    b_model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        tqdm_dataloader = tqdm(dataloader)
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, candidates, labels = data
            policy_action = p_model(seqs)
            scores = b_model.predict(seqs, policy_action, candidates)
            #
            metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
    print(avg_metrics)
    for k in sorted(args.metric_ks, reverse=True):
        writer.add_scalar('Train/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
    return avg_metrics

def ProfileTrain(epochs, model, train_loader, val_loader, args):
    if args.is_pretrain == 0:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(args.device)
    best_acc = 0
    for epoch in range(epochs):
        ProfileTrainer(epoch, model, train_loader, optimizer, args)
        acc = ProfileValidate(epoch, model, val_loader, args)
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(args.save_path,
                                                '{}_{}_seed{}_profile-{}_pretrain{}_best_model.pth'.format(args.task_name,
                                                                                     args.model_name, args.seed, args.user_profile, args.is_pretrain)))

def ProfileTrainer(epoch, model, dataloader, optimizer, args):
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    running_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data  # , _, _
        logits = model(seqs)  # B x T x V
        labels = labels.view(-1)  # B*T
        logits = logits.mean(1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(logits, labels)
        running_loss += loss.detach().cpu().item()
        running_acc += acc
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))

def ProfileValidate(epoch, model, dataloader, args):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0
    running_acc = 0
    with torch.no_grad():
        tqdm_dataloader = dataloader
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            logits = model(seqs)
            labels = labels.view(-1)  # B*T
            logits = logits.mean(1)
            loss = loss_fn(logits, labels)
            acc = accuracy(logits, labels)
            running_loss += loss
            running_acc += acc
    print("Validation CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    print("Validation accuracy: {:.5f}".format(running_acc / len(dataloader)))
    avg_acc = running_acc / len(dataloader)
    return avg_acc

def accuracy(logits, labels):
    predicts = torch.max(logits, 1)[1]
    equal = labels.eq(predicts)
    acc = equal.sum()/len(labels)
    return acc

def SeqTrain(epochs, model, train_loader, val_loader, writer, args):
    if args.is_pretrain == 0:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(args.device)
    if args.is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(model,  find_unused_parameters=True,device_ids=[args.local_rank], output_device=args.local_rank)
    best_metric = 0
    all_time = 0
    val_all_time = 0
    for epoch in range(epochs):
        since = time.time()
        optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)
        tmp = time.time() - since
        print('one epoch train:', tmp)
        all_time += tmp
        val_since = time.time()
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)
        val_tmp = time.time() - val_since
        print('one epoch val:', val_tmp)
        val_all_time += val_tmp
        if args.is_pretrain == 0 and 'acc' in args.task_name:
            if metrics['NDCG@20'] >= 0.0193:
                break
        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
            state_dict = model.state_dict()
            if 'life' in args.task_name:
                torch.save(state_dict, os.path.join(args.save_path,
                                                         '{}_{}_seed{}_task_{}_best_model.pth'.format('sequence',
                                                                                                      args.model_name,
                                                                                                      args.seed,
                                                                                                      args.task)))
            else:
                torch.save(state_dict, os.path.join(args.save_path, '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_model



def SequenceTrainer(epoch, model, dataloader, optimizer, writer, args): #schedular,
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data
        logits = model(seqs) # B x T x V
        if 'cold' in args.task_name or ('life_long' in args.task_name and args.task != 0):
            logits = logits.mean(1)
            labels = labels.view(-1)
        else:
            logits = logits.view(-1, logits.size(-1)) # (B*T) x V
            labels = labels.view(-1)  # B*T

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    return optimizer

def Sequence_pn_Trainer(epoch, model, dataloader, optimizer, writer, args):  # schedular,
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, pos, neg = data  # , _, _
        pos_logits, neg_logits = model(seqs, pos, neg)
        pos_labels = torch.ones(pos_logits.shape, device=args.device)
        neg_labels = torch.zeros(neg_logits.shape, device=args.device)
        indices = torch.where(pos != 0)
        loss = loss_fn(pos_logits[indices], pos_labels[indices])
        loss += loss_fn(neg_logits[indices], neg_labels[indices])
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    return optimizer

def Sequence_full_Validate(epoch, model, dataloader, writer, args, test=False):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        tqdm_dataloader = dataloader
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            if test:
                scores = model.predict(seqs)
            else:
                scores = model(seqs)
            if 'cold' in args.task_name or ('life_long' in args.task_name and args.task != 0):
                scores = scores.mean(1)
            else:
                scores = scores[:, -1, :]
            metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
    print(avg_metrics)
    for k in sorted(args.metric_ks, reverse=True):
        writer.add_scalar('Train/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
    return avg_metrics

def Sequence_neg_Validate(epoch, model, optimizer, dataloader, writer, args):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        tqdm_dataloader = dataloader
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, candidates, labels = data
            scores = model.predict(seqs, candidates)  #
            metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
    print(avg_metrics)
    for k in sorted(args.metric_ks, reverse=True):
        writer.add_scalar('Train/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
    return avg_metrics

def KDTrain(epochs, teacher_model, student_model, train_loader, val_loader, writer, args): #, user_noclicks
    if args.task_name == 'transfer_learning' and args.is_pretrain == 0:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), betas=(0.9, 0.98), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_model = teacher_model.to(args.device)
    student_model = student_model.to(args.device)
    best_metric = 0
    for epoch in range(epochs):
        optimizer = KDTrainer(epoch, teacher_model, student_model, train_loader, optimizer, writer, args)#scheduler,
        metrics = Sequence_full_Validate(epoch, student_model, val_loader, writer, args)

        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            state_dict = student_model.state_dict()
            torch.save(state_dict, os.path.join(args.save_path,
                                                '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(
                                                    args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                    args.lr, args.weight_decay, args.block_num, args.hidden_size,
                                                    args.embedding_size)))

def KDTrainer(epoch, teacher_model, student_model, dataloader, optimizer, writer, args): #schedular,
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    student_model.train()
    teacher_model.eval()
    running_loss = 0
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_token)
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    temp = args.temp
    alpha = args.alpha
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data #, _, _
        with torch.no_grad():
            teacher_pred = teacher_model(seqs)
        logits = student_model(seqs) # B x T x V
        teacher_pred = teacher_pred.view(-1, teacher_pred.size(-1))
        logits = logits.view(-1, logits.size(-1)) # (B*T) x V
        labels = labels.view(-1)  # B*T
        ce_loss = ce_loss_fn(logits, labels)
        distillation_loss = kd_loss_fn(
            F.softmax(logits / temp, dim=1),
            F.softmax(teacher_pred / temp, dim=1)
        )
        loss = alpha * ce_loss + (1 - alpha) * distillation_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    return optimizer

def recalls_and_ndcgs_for_ks(scores, labels, ks, args):
    metrics = {}

    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float()).to(args.device)
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(args.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg

    return metrics

def _create_state_dict(model, optimizer, args):
    return {
        'model_state_dict': model.module.state_dict() if args.is_parallel else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

def lifelong_Train(epochs, model, train_loader, val_loader, writer, task_times, args): #, user_noclicks
    print("++++++++train_with_prune_task{}++++++++".format(args.task))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), lr=args.lr,
                                 weight_decay=args.weight_decay)
    model = model.to(args.device)
    if task_times != 0:
        if task_times == 1:
            current_mask = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 1)),
                                      map_location=torch.device('cuda'))
            current_mask = reverse_mask(current_mask)
        elif task_times == 2:
            current_mask1 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 1)),
                                      map_location=torch.device('cuda'))
            current_mask2 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 2)),
                                       map_location=torch.device('cuda'))
            current_mask = concat_mask(current_mask1, current_mask2)
            current_mask = reverse_mask(current_mask)
        elif task_times == 3:
            current_mask1 = torch.load(
                os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 3)),
                map_location=torch.device('cuda'))
            current_mask2 = torch.load(
                os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 2)),
                map_location=torch.device('cuda'))
            current_mask3 = torch.load(
                os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 1)),
                map_location=torch.device('cuda'))
            current_mask = concat_mask(current_mask1, current_mask2)
            current_mask = concat_mask(current_mask, current_mask3)
            current_mask = reverse_mask(current_mask)

        model_path = os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times-1))
        best_weight = torch.load(model_path, map_location=torch.device(args.device))
        best_weight = reverse_rewind(best_weight)
        model.load_state_dict(best_weight)
        check_sparsity_dict(current_mask)
    best_metric = 0
    for epoch in range(epochs):
        if task_times == 0:
            optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)  # scheduler,
        else:
            lifelongTrainer(epoch, model, current_mask, train_loader, optimizer, writer, args)
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)

    # current_metric = metrics['ndcg_20']
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
            save_model_dict = deepcopy(best_model.state_dict())
            torch.save(save_model_dict, os.path.join(args.save_path,
                                                         '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name,
                                                                                                      args.model_name,
                                                                                                      args.seed, task_times)))
            # state_dict = _create_state_dict(model, optimizer, args)
            # torch.save(state_dict, os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times)))

    if task_times != 0:
        prune_model_custom(best_model, current_mask, args)
        check_sparsity(best_model, args)
    #after training
    if task_times != args.task_num - 1:
        # save_model_dict = deepcopy(best_model.state_dict())
        pruning_model(best_model, args.prun_rate, args)
        model_dict = best_model.state_dict()
        check_sparsity(best_model, args)
        # torch.save(save_model_dict, os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times)))
        mask_dict = extract_mask(model_dict)
        check_sparsity_dict(mask_dict)
        torch.save(mask_dict, os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times)))
    # else:
    #     # save_model_dict = best_model.state_dict()
    #     save_model_dict = reverse_rewind(save_model_dict)
    #     torch.save(save_model_dict, os.path.join(args.save_path,
    #                                              '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name,
    #                                                                                           args.model_name,
    #                                                                                           args.seed, task_times)))


def lifelong_ReTrain(epochs, model, train_loader, val_loader, test_loader, writer, task_times, args): #, user_noclicks
    print("++++++++retrain_task{}++++++++".format(task_times))
    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= args.decay_step, gamma=args.gamma)
    model = model.to(args.device)
    if task_times == 0:
        current_mask2 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times)),
                                  map_location=torch.device('cuda'))
        current_mask = current_mask2
    elif task_times == 1:
        current_mask1 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times-1)),
                                  map_location=torch.device('cuda'))
        current_mask2 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times)),
                                  map_location=torch.device('cuda'))
        # current_mask2 = reverse_mask(current_mask2)
        current_mask = concat_mask(current_mask1, current_mask2)
        # current_mask = reverse_mask(current_mask)
    else:
        current_mask0 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 2)),
                                   map_location=torch.device('cuda'))
        current_mask1 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times - 1)),
                                   map_location=torch.device('cuda'))
        current_mask2 = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times)),
                                   map_location=torch.device('cuda'))
        # current_mask2 = reverse_mask(current_mask2)
        current_mask = concat_mask(current_mask0, current_mask1)
        current_mask = concat_mask(current_mask, current_mask2)
    model_path = os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times))
    new_dict = torch.load(model_path, map_location=torch.device(args.device))
    model.load_state_dict(new_dict)
    check_sparsity_dict(current_mask)
    # prune_model_custom(model, current_mask, args)
    # check_sparsity(model, args)
    best_metric = 0
    model_mask(model, current_mask, args)

    for epoch in range(epochs):
        if task_times == 0:
            optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)#scheduler,
        else:
            lifelongTrainer(epoch, model, current_mask2, train_loader, optimizer, writer, args)
        # running_loss, length = SequenceTrainer(epoch, model, train_loader, optimizer, scheduler, args)
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)

        # current_metric = metrics['ndcg_20']
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
    #test
    Sequence_full_Validate(0, best_model, test_loader, writer, args)
    retain_mask(best_model, new_dict, args)
    state_dict = best_model.state_dict()
    # state_dict = reverse_rewind(state_dict)
    # remove_prune(best_model)
    # state_dict = best_model.state_dict()#_create_state_dict(model, optimizer, args)
    torch.save(state_dict, os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times)))

def lifelong_ReTrain1(epochs, model, train_loader, val_loader, test_loader, writer, task_times, args): #, user_noclicks
    print("++++++++retrain_task{}++++++++".format(task_times))
    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= args.decay_step, gamma=args.gamma)
    model = model.to(args.device)
    current_mask = torch.load(os.path.join(args.save_path, 'seed{}_{}_task_mask_{}.pth'.format(args.seed, args.model_name, task_times)),
                              map_location=torch.device('cuda'))

    model_path = os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, 3))
    new_dict = torch.load(model_path, map_location=torch.device(args.device))
    model.load_state_dict(new_dict)
    check_sparsity_dict(current_mask)
    prune_model_custom(model, current_mask, args)
    check_sparsity(model, args)
    best_metric = 0
    # model_mask(model, current_mask, args)

    for epoch in range(epochs):
        # if task_times == 0:
        optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)#scheduler,
        # else:
        #     lifelongTrainer(epoch, model, current_mask2, train_loader, optimizer, writer, args)
        # running_loss, length = SequenceTrainer(epoch, model, train_loader, optimizer, scheduler, args)
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)

        # current_metric = metrics['ndcg_20']
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
    #test
    Sequence_full_Validate(0, best_model, test_loader, writer, args)
    # retain_mask(best_model, new_dict, args)
    state_dict = best_model.state_dict()
    state_dict = reverse_rewind(state_dict)
    # remove_prune(best_model)
    # state_dict = best_model.state_dict()#_create_state_dict(model, optimizer, args)
    torch.save(state_dict, os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, 3)))

# def lifelong_ReTrain(epochs, model, train_loader, val_loader, writer, task_times, args): #, user_noclicks
#     print("++++++++retrain_task{}++++++++".format(args.task))
#     optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=args.lr, weight_decay=args.weight_decay)
#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= args.decay_step, gamma=args.gamma)
#     model = model.to(args.device)
#     current_mask = torch.load(os.path.join(args.save_path, '{}_task_mask_{}.pth'.format(args.model_name, task_times-1)),
#                               map_location=torch.device('cuda'))
#
#     model_path = os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times))
#     new_dict = torch.load(model_path, map_location=torch.device(args.device))
#     model.load_state_dict(new_dict)
#     prune_model_custom(model, current_mask, args)
#     check_sparsity(model, args)
#     best_metric = 0
#     for epoch in range(epochs):
#         # optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)#scheduler,
#         if task_times == 0:
#             optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)  # scheduler,
#         else:
#             lifelongTrainer(epoch, model, current_mask, train_loader, optimizer, writer, args)
#         # running_loss, length = SequenceTrainer(epoch, model, train_loader, optimizer, scheduler, args)
#         # if (epoch + 1) % 5 == 0:
#         metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)
#
#     # current_metric = metrics['ndcg_20']
#         current_metric = metrics['NDCG@5']
#         if best_metric <= current_metric:
#             best_metric = current_metric
#             best_model = deepcopy(model)
#     state_dict = best_model.state_dict()
#     state_dict = reverse_rewind(state_dict)
#     # remove_prune(best_model)
#     # state_dict = best_model.state_dict()#_create_state_dict(model, optimizer, args)
#     torch.save(state_dict, os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times)))

    #after training
    # if task_times != 2:
    #     save_model_dict = deepcopy(best_model.state_dict())
    #     pruning_model(best_model, args.purn_rate, args)
    #     check_sparsity(best_model, args)
    #     torch.save(save_model_dict, os.path.join(args.save_path, '{}_{}_seed{}_task_{}_best_model.pth'.format(args.task_name, args.model_name, args.seed, task_times-1)))
    #     mask_dict = extract_mask(save_model_dict)
    #     check_sparsity_dict(mask_dict)
    #     torch.save(mask_dict, os.path.join(args.save_path, 'task_mask_{}.pth'.format(task_times)))

def lifelongTrainer(epoch, model, mask, dataloader, optimizer, writer, args): #schedular,
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    length = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_token)
    # loss_fn = nn.BCEWithLogitsLoss()
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data  # , _, _
        logits = model(seqs)  # B x T x V
        # parm_dict = {}
        # for name, parm in model.named_parameters():
        #     parm_dict[name] = parm

        logits = logits.mean(1)
        labels = labels.view(-1)
        # logits = logits.mean(1)
        # state = deepcopy(model.state_dict())
        loss = loss_fn(logits, labels)
        loss.backward()
        grad_mask(model, mask, args)
        optimizer.step()
        # new_state = model.state_dict()
        # schedular.step()
        running_loss += loss.detach().cpu().item()
    # schedular.step()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    return optimizer

def model_mask(model, mask, args):
    if 'bert' in args.model_name:
        for name, m in model.named_modules():  # model.modules()module.
            if (isinstance(m, nn.Linear) and 'out' not in name) or isinstance(m, nn.LayerNorm): # or isinstance(m, nn.Embedding):
                # if isinstance(m, nn.LayerNorm):
                #     tmp_mask = mask[name + '.weight_mask']
                #     # try:
                #     #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                #     # except:
                #     m.weight.data = m.weight.data * tmp_mask
                # else:
                tmp_w_mask = mask[name + '.weight_mask']
                tmp_b_mask = mask[name + '.bias_mask']
                m.weight.data = m.weight.data * tmp_w_mask
                m.bias.data = m.bias.data * tmp_b_mask
                # m.weight

    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):# or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                tmp_w_mask = mask[name + '.weight_mask']
                tmp_b_mask = mask[name + '.bias_mask']
                # try:
                #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # except:
                m.weight.data = m.weight.data * tmp_w_mask
                m.bias.data = m.bias.data * tmp_b_mask

def retain_mask(model, state, args):
    if 'bert' in args.model_name:
        for name, m in model.named_modules():  # model.modules()module.
            if (isinstance(m, nn.Linear) and 'out' in name) or isinstance(m, nn.LayerNorm): # or isinstance(m, nn.Embedding):
                # if isinstance(m, nn.LayerNorm):
                #     tmp_state = state[name + '.weight']
                #     # try:
                #     #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                #     # except:
                #     m.weight.data = torch.where(m.weight.data>0, m.weight.data, tmp_state)
                # else:
                tmp_w_state = state[name + '.weight']
                tmp_b_state = state[name + '.bias']
                # try:
                #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # except:
                m.weight.data = torch.where(m.weight.data > 0, m.weight.data, tmp_w_state)
                m.bias.data = torch.where(m.bias.data > 0, m.bias.data, tmp_b_state)

    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):# or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):

                tmp_w_state = state[name + '.weight']
                tmp_b_state = state[name + '.bias']
                # try:
                #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # except:
                m.weight.data = torch.where(m.weight.data > 0, m.weight.data, tmp_w_state)
                m.bias.data = torch.where(m.bias.data > 0, m.bias.data, tmp_b_state)

def grad_mask(model, mask, args):
    if 'bert' in args.model_name:
        for name, m in model.named_modules():  # model.modules()module.
            if (isinstance(m, nn.Linear) and 'out' not in name) or isinstance(m, nn.LayerNorm):# or (isinstance(m, nn.Embedding) and 'token' in name):
                # if isinstance(m, nn.LayerNorm):
                #     tmp_mask = mask[name + '.weight_mask']
                # # try:
                # #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # # except:
                #     m.weight.grad = m.weight.grad * tmp_mask
                # else:
                tmp_w_mask = mask[name + '.weight_mask']
                tmp_b_mask = mask[name + '.bias_mask']
                # try:
                #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # except:
                m.weight.grad = m.weight.grad * tmp_w_mask
                m.bias.grad = m.bias.grad * tmp_b_mask

    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):  # or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                # tmp_mask = mask[name + '.weight_mask']
                # try:
                #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # except:
                #     m.weight.grad = m.weight.grad * tmp_mask
                tmp_w_mask = mask[name + '.weight_mask']
                tmp_b_mask = mask[name + '.bias_mask']
                # try:
                #     m.weight_orig.grad = m.weight_orig.grad * tmp_mask
                # except:
                m.weight.grad = m.weight.grad * tmp_w_mask
                m.bias.grad = m.bias.grad * tmp_b_mask

def reverse_rewind(model_dict):
    out_dict = {}
    for key in model_dict.keys():
        if 'orig' in key:
            out_dict[key[:-5]] = model_dict[key]
        else:
            out_dict[key] = model_dict[key]
        if 'mask' in key:
            out_dict.pop(key)
    return out_dict

def remove_prune(model, args):
    if 'bert' in args.model_name:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Linear):# or (isinstance(m, nn.Embedding) and 'token' in name):
                prune.remove(m,'weight')
    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or ((isinstance(m, nn.Linear)) and 'final_layer' in name):# or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                prune.remove(m, 'weight')

def concat_mask(mask1,mask2):
    comask = {}
    for key in mask1.keys():
        comask[key] = mask1[key] + mask2[key]

    return comask

def reverse_mask(orig_mask):
    remask = {}
    for key in orig_mask.keys():
        remask[key] = 1-orig_mask[key]

    return remask

def prune_model_custom(model, mask_dict, args):
    # print(mask_dict)
    if 'bert' in args.model_name:
        for name, m in model.named_modules() :  # model.modules()module.
            if (isinstance(m, nn.Linear) and 'out' not in name) or isinstance(m, nn.LayerNorm): # or isinstance(m, nn.Embedding):
                # if isinstance(m, nn.LayerNorm):
                #     prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name + '.weight_mask'])
                # else:
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])#module. 'bert.'
                prune.CustomFromMask.apply(m, 'bias', mask=mask_dict[name + '.bias_mask'])

    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):# or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name + '.weight_mask']) #'residual_blocks.' +
                prune.CustomFromMask.apply(m, 'bias', mask=mask_dict[name + '.bias_mask'])

def re_weight(model, args):
    new_dict = {}
    state_dict = model.state_dict()
    if 'bert' in args.model_name:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Linear):# or (isinstance(m, nn.Embedding) and 'token' in name):
                key_orig = name + '.weight_orig'
                key = name + '.weight'
                new_dict[key] = state_dict[key_orig]
    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or ((isinstance(m, nn.Linear)) and 'final_layer' in name):# or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                key_orig = name + '.weight_orig'
                key = name + '.weight'
                new_dict[key] = state_dict[key_orig]
    return new_dict

def check_sparsity_dict(model_dict):
    sum_list = 0
    zero_sum = 0
    for key in model_dict.keys():
        sum_list = sum_list+float(model_dict[key].nelement())
        zero_sum = zero_sum+float(torch.sum(model_dict[key] == 0))

    print('* remain_weight = {:.4f} %'.format(100*(1-zero_sum/sum_list)))

    return 100*(1-zero_sum/sum_list)

def extract_mask(model_dict):
    mask_weight = {}
    for key in model_dict.keys():
        if 'mask' in key:
            mask_weight[key] = model_dict[key]

    return mask_weight

def check_sparsity(model, args):
    sum_list = 0
    zero_sum = 0
    if 'bert' in args.model_name:
        for name, m in model.named_modules():  # model.modules()module.
            if (isinstance(m, nn.Linear) and 'out' not in name) or isinstance(m, nn.LayerNorm):  # or isinstance(m, nn.Embedding):
                # if isinstance(m, nn.LayerNorm):
                #     sum_list = sum_list + float(m.weight.nelement())
                #     zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                # else:
                sum_list = sum_list + float(m.weight.nelement())
                sum_list = sum_list + float(m.bias.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                zero_sum = zero_sum + float(torch.sum(m.bias == 0))
    else:
        for name, m in model.named_modules():  # model.modules()module.
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):# or isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                sum_list = sum_list + float(m.weight.nelement())
                sum_list = sum_list + float(m.bias.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                zero_sum = zero_sum + float(torch.sum(m.bias == 0))

    print('* remain_weight = {:.4f} %'.format(100*(1-zero_sum/sum_list)))

    return 100*(1-zero_sum/sum_list)

def pruning_model(model, px, args):

    parameters_to_prune = []
    if 'bert' in args.model_name:
        for name, m in model.named_modules(): #model.modules()module.
            if (isinstance(m, nn.Linear) and 'out' not in name)  or isinstance(m, nn.LayerNorm):
                # parameters_to_prune.append((m, 'weight'))
                # if isinstance(m, nn.LayerNorm):
                #     prune.L1Unstructured.apply(m, name='weight', amount=px)
                # else:
                prune.L1Unstructured.apply(m, name='weight', amount=px)
                prune.L1Unstructured.apply(m, name='bias', amount=px)
    else:
        for name, m in model.named_modules():# model.modules()module.
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):# or isinstance(m, nn.Embedding):
                # parameters_to_prune.append((m, 'weight'))
                prune.L1Unstructured.apply(m, name='weight', amount=px)
                prune.L1Unstructured.apply(m, name='bias', amount=px)
    # parameters_to_prune = tuple(parameters_to_prune)
    #
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=px,
    # )
