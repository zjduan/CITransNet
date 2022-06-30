import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer2DNet(nn.Module):
    def __init__(self, d_input, d_output, n_layer, n_head, d_hidden):
        super(Transformer2DNet, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_layer = n_layer

        d_in = d_input
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(n_layer):
            d_out = d_hidden if i != n_layer - 1 else d_output
            self.row_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.col_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.fc.append(nn.Sequential(
                nn.Linear(d_in + 2 * d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_out)
            ))
            d_in = d_hidden

    def forward(self, input):
        bs, n_bidder, n_item, d = input.shape
        x = input
        for i in range(self.n_layer):
            row_x = x.view(-1, n_item, d)
            row = self.row_transformer[i](row_x)
            row = row.view(bs, n_bidder, n_item, -1)

            col_x = x.permute(0, 2, 1, 3).reshape(-1, n_bidder, d)
            col = self.col_transformer[i](col_x)
            col = col.view(bs, n_item, n_bidder, -1).permute(0, 2, 1, 3)

            glo = x.view(bs, n_bidder*n_item, -1).mean(1, keepdim=True)
            glo = glo.unsqueeze(1) # (bs, 1, 1, -1)
            glo = glo.repeat(1, n_bidder, n_item, 1)

            x = torch.cat([row, col, glo], dim=-1)
            x = self.fc[i](x)
        return x

class TransformerMechanism(nn.Module):
    def __init__(self, n_bidder_type, n_item_type, d_emb, n_layer, n_head, d_hidden, v_min=0, v_max=1,
                 continuous_context=False, cond_prob=False):
        super(TransformerMechanism, self).__init__()
        self.d_emb = d_emb
        self.v_min = v_min
        self.v_max = v_max
        self.continuous_context = continuous_context
        self.pre_net = nn.Sequential(
            nn.Linear(d_emb*2+1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden-1)
        )
        d_input = d_hidden
        self.n_layer, self.n_head, self.d_hidden =  n_layer, n_head, d_hidden
        self.mechanism = Transformer2DNet(d_input, 3, self.n_layer, n_head, d_hidden)
        if not continuous_context:
            self.bidder_embeddings = nn.Embedding(n_bidder_type, d_emb)
            self.item_embeddings = nn.Embedding(n_item_type, d_emb)
        self.cond_prob = cond_prob

    def forward(self, batch_data):
        raw_bid, bidder_context, item_context = batch_data
        bid = (raw_bid - self.v_min) / (self.v_max - self.v_min)
        bs, n_bidder, n_item = bid.shape
        x1 = bid.unsqueeze(-1) # (bs, n, m, 1)

        if self.continuous_context:
            bidder_emb = bidder_context
            item_emb = item_context
        else:
            bidder_emb = self.bidder_embeddings(bidder_context.view(-1, n_bidder))
            item_emb = self.item_embeddings(item_context.view(-1, n_item))

        x2 = bidder_emb.unsqueeze(2).repeat(1, 1, n_item, 1)
        x3 = item_emb.unsqueeze(1).repeat(1, n_bidder, 1, 1)

        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.pre_net(x)
        x = torch.cat([x1, x], dim=-1)

        mechanism = self.mechanism(x)
        allocation, allocation_prob, payment = \
            mechanism[:, :, :, 0], mechanism[:, :, :, 1], mechanism[:, :, :, 2] # (bs, n, m)

        allocation = F.softmax(allocation, dim=1)
        if self.cond_prob:
            allocation_prob = allocation_prob.mean(-2, keepdims=True)
        allocation_prob = torch.sigmoid(allocation_prob)
        allocation = allocation * allocation_prob

        payment = payment.mean(-1)
        payment = torch.sigmoid(payment)
        payment = (raw_bid * allocation).sum(-1) * payment

        return allocation, payment
