import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import StarAggregator, GlobalAggregator, LastAggregator, SessionAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.hop = opt.n_iter
        self.step = opt.step
        self.dropout_global = opt.dropout_global
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.scale = opt.scale
        self.norm = opt.norm
        self.tau = opt.tau
        self.lam = opt.lambda_
        self.last_num = opt.last_len

        # Aggregator
        self.star_agg = StarAggregator(self.dim, self.opt.alpha, self.step)
        self.global_agg = GlobalAggregator(self.dim, self.hop, self.sample_num, opt.dropout_gcn)
        self.last_agg = LastAggregator(self.dim)
        self.sess_agg = SessionAggregator(self.dim)

        # Parameter
        self.s = nn.Parameter(torch.Tensor(2, 1))

        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim)

        self.soft_func = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, last_hidden, mask, targets, is_test):
        mask = mask.float().unsqueeze(-1)

        select = self.sess_agg(hidden, mask)
        last_select = last_hidden.squeeze()

        b = self.embedding.weight[1:]  # n_nodes x latent_size

        if self.norm:
            select = F.normalize(select, p=2, dim=-1)
            last_select = F.normalize(last_select, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)

        scores = torch.matmul(select, b.transpose(1, 0))
        last_scores = torch.matmul(last_select, b.transpose(1, 0))

        if self.scale:
            scores = self.tau * scores  # tau is the sigma factor
            last_scores = self.tau * last_scores

        if is_test:
            main_output = self.soft_func(scores)
            last_output = self.soft_func(last_scores)
        else:
            targets = trans_to_cuda(targets).long()
            main_output = self.loss_func(scores, targets - 1)
            last_output = self.loss_func(last_scores, targets - 1)
        
        output = main_output + self.lam * last_output
        
        return output

    def forward(self, inputs, adj, item, last_items, adj_items):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        mask = torch.ne(inputs, trans_to_cuda(torch.zeros(inputs.shape).long())).float()
        h_local, _ = self.star_agg(h, adj, mask)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        h_global = self.global_agg(entity_vectors[0], entity_vectors, weight_neighbors)

        h_local = F.normalize(h_local, p=2, dim=-1)
        h_global = F.normalize(h_global, p=2, dim=-1)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = torch.matmul(torch.stack([h_local, h_global], dim=-1), self.s).squeeze()

        # short-term
        # node embedding
        item_hidden = self.embedding(last_items)
        adj_hidden = self.embedding(adj_items)
        adj_hidden = adj_hidden.view(adj_hidden.shape[0], self.last_num, -1, self.dim)
        # adj info aggregation
        last_hidden = self.last_agg(item_hidden, adj_hidden)

        return output, last_hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, is_test=False):
    alias_inputs, adj, items, mask, targets, inputs, last_items, adj_items = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    last_items = trans_to_cuda(last_items).long()
    adj_items = trans_to_cuda(adj_items).long()

    hidden, last_hidden = model(items, adj, inputs, last_items, adj_items)

    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    return targets, model.compute_scores(seq_hidden, last_hidden, mask, targets, is_test)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, loss = forward(model, data)
        # targets = trans_to_cuda(targets).long()
        # loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit_10, hit_20, mrr_10, mrr_20 = [], [], [], []
    for data in test_loader:
        targets, scores = forward(model, data, is_test=True)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit_10) * 100)
    result.append(np.mean(hit_20) * 100)
    result.append(np.mean(mrr_10) * 100)
    result.append(np.mean(mrr_20) * 100)

    return result
