import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import math


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class StarAggregator(nn.Module):
    def __init__(self, dim, alpha, step, name=None):
        super(StarAggregator, self).__init__()
        self.dim = dim
        self.agg = LocalAggregator(self.dim, alpha, dropout=0.0)
        self.step = step

        self.q_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.k_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.q_2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.k_2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.linear_weight = nn.Linear(self.dim * 2, self.dim, bias=False)

    def mask_softmax(self, logits, mask):
        mask_bool = (mask == 0)
        logits[mask_bool] = float('-inf')
        return torch.softmax(logits, -1)

    def forward(self, hidden, adj, mask):
        avg_hidden = torch.sum(hidden * mask.unsqueeze(-1), 1) / torch.sum(mask, 1, keepdim=True)
        star_hidden = avg_hidden.unsqueeze(1)
        sate_hidden = hidden

        for i in range(self.step):
            # print(inputs.shape, hidden.shape, star_hidden.shape)

            sate_hidden = self.agg(sate_hidden, adj)

            # satellite nodes update
            query = torch.matmul(sate_hidden, self.q_1)
            key = torch.matmul(star_hidden, self.k_1)
            alpha = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            
            sate_hidden = (1. - alpha) * sate_hidden + alpha * star_hidden

            # print("sate hidden:\n", sate_hidden.shape)
            
            # star node update
            query = torch.matmul(star_hidden, self.q_2)
            key = torch.matmul(sate_hidden, self.k_2)
            beta = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            # beta = F.softmax(beta, dim=-1) * mask.unsqueeze(1)
            beta = self.mask_softmax(beta, mask.unsqueeze(1))
            star_hidden = torch.matmul(beta, sate_hidden)

            # print("star hiddden:\n", star_hidden.shape)
        
        weight = torch.sigmoid(self.linear_weight(torch.cat([hidden, sate_hidden], dim=-1)))
        output_hidden = weight * hidden + (1 - weight) * sate_hidden

        return output_hidden, star_hidden


class GlobalAggregator(nn.Module):
    def __init__(self, dim, hop, sample_num, dropout, name=None):
        super(GlobalAggregator, self).__init__()
        self.dim = dim
        self.hop = hop
        self.sample_num = sample_num
        self.dropout = dropout

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_4 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.s = nn.Parameter(torch.Tensor(self.hop, 1))

    def aggregate(self, item_hidden, neighbor_hidden, neighbor_weight):
        batch_size = item_hidden.shape[0]

        alpha = torch.matmul(torch.cat([item_hidden.unsqueeze(2).repeat(1, 1, neighbor_hidden.shape[2], 1) * neighbor_hidden, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        output = torch.sum(alpha * neighbor_hidden, dim=-2)
        # neighbor_hidden = torch.mean(neighbor_hidden, dim=2)
            
        output = F.dropout(output, self.dropout, training=self.training)
        output = output.view(batch_size, -1, self.dim)
        return output
 
    def forward(self, hidden, neighbor_hiddens, neighbor_weights):
        batch_size = hidden.shape[0]

        agg_hiddens = []
        for i in range(self.hop):
            next_neighbor_hiddens = []
            hidden_shape = [batch_size, -1, self.sample_num, self.dim]
            weight_shape = [batch_size, -1, self.sample_num]
            for j in range(self.hop - i):
                agg_hidden = self.aggregate(neighbor_hiddens[j], neighbor_hiddens[j + 1].view(hidden_shape), neighbor_weights[j].view(weight_shape))
                next_neighbor_hiddens.append(agg_hidden)
            neighbor_hiddens = next_neighbor_hiddens
            agg_hiddens.append(neighbor_hiddens[0].view(hidden.shape))

        agg_hidden = torch.matmul(torch.stack(agg_hiddens, dim=-1), self.s).squeeze(-1)
        weight = torch.sigmoid(torch.matmul(hidden, self.w_3) + torch.matmul(agg_hidden, self.w_4))
        output = (1 - weight) * hidden + weight * agg_hidden

        return output


class SessionAggregator(nn.Module):
    def __init__(self, dim, name=None):
        super(SessionAggregator, self).__init__()
        self.dim = dim

        # Position representation
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.linear_one = nn.Linear(self.dim, self.dim)
        self.linear_two = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, hidden, mask):
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_hidden = self.pos_embedding.weight[:len]
        pos_hidden = pos_hidden.unsqueeze(0).repeat(batch_size, 1, 1)

        sess_h = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        sess_h = sess_h.unsqueeze(-2).repeat(1, hidden.shape[1], 1)
        
        pos_h = torch.matmul(torch.cat([pos_hidden, hidden], -1), self.w_1)
        pos_h = torch.tanh(pos_h)
        
        alpha = torch.sigmoid(self.linear_one(pos_h) + self.linear_two(sess_h))
        alpha = torch.matmul(alpha, self.w_2)
        output = torch.sum(alpha * mask * hidden, 1)

        return output


class LastAggregator(nn.Module):
    def __init__(self, dim, dropout=0.4, name=None):
        super(LastAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.q_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.k_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.linear_one = nn.Linear(self.dim, self.dim)
        self.linear_two = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_weight = nn.Linear(self.dim, 1, bias=False)

    def forward(self, item_hidden, adj_hidden):
        # calculate importance
        query = torch.matmul(adj_hidden, self.q_1)   # b x n x m x d
        key = torch.matmul(item_hidden.unsqueeze(-2), self.k_1)    # b x n x 1 x d
        alpha = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)    # b x n x m x 1
        # adj info aggregation
        agg_hidden = torch.sum(alpha * adj_hidden, dim=-2)   # b x n x d
        agg_hidden = F.dropout(agg_hidden, self.dropout, training=self.training)
        weight = torch.sigmoid(torch.matmul(item_hidden, self.w_1) + torch.matmul(agg_hidden, self.w_2))    # b x n x d
        final_hidden = (1 - weight) * item_hidden + weight * agg_hidden  # b x n x d
        # generate short-term preference
        avg_hidden = torch.sum(final_hidden, dim=1, keepdim=True) / final_hidden.shape[1]  # b x 1 x d
        beta = self.linear_weight(torch.sigmoid(self.linear_one(final_hidden) + self.linear_two(avg_hidden))) # b x n x 1
        last_hidden = torch.sum(beta * final_hidden, dim=1)    # b x d
        return last_hidden
