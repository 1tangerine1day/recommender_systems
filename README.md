# recommender systems algorithm
* Matrix Factorization
* Bayesian Personalized Ranking


## model
* MF

        class MF(nn.Module):
        def __init__(self, num_factors, num_users, num_items, **kwargs):
            super().__init__()
            self.P = nn.Embedding(num_users, num_factors)
            self.Q = nn.Embedding(num_items, num_factors)
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)

        def forward(self, user_id, item_id):
            P_u = self.P(user_id)
            Q_i = self.Q(item_id)
            b_u = self.user_bias(user_id)
            b_i = self.item_bias(item_id)

            outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
            outputs =  outputs.flatten()

            return outputs

        def recommend(self, x, y):
            user_id = torch.Tensor(np.array([x])).type(torch.LongTensor).cuda()
            item_id = torch.Tensor(np.array([y])).type(torch.LongTensor).cuda()
            P_u = self.P(user_id)
            Q_i = self.Q(item_id)
            b_u = self.user_bias(user_id)
            b_i = self.item_bias(item_id)

            outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
            outputs =  outputs.flatten()

            return outputs
            
* BPR

        class BPR(nn.Module):
            def __init__(self, user_size, item_size, dim, weight_decay):
                super().__init__()

                self.W = nn.Parameter(torch.empty(user_size, dim))
                self.H = nn.Parameter(torch.empty(item_size, dim))
                nn.init.xavier_normal_(self.W.data)
                nn.init.xavier_normal_(self.H.data)
                self.weight_decay = weight_decay

            def forward(self, u, i, j):

                u = self.W[u, :]
                i = self.H[i, :]
                j = self.H[j, :]
                x_ui = torch.mul(u, i).sum(dim=1)
                x_uj = torch.mul(u, j).sum(dim=1)
                x_uij = x_ui - x_uj
                log_prob = F.logsigmoid(x_uij).sum()
                regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
                return -log_prob + regularization

            def recommend(self, u):

                u = self.W[u, :]
                x_ui = torch.mm(u, self.H.t())
                pred = torch.argsort(x_ui, dim=1)
                return pred
## data

MovieLens 100K Dataset : https://grouplens.org/datasets/movielens/
