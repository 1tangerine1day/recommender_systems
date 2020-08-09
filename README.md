# recommender systems algorithm
* Matrix Factorization
* Bayesian Personalized Ranking
* Deep Factorization Machines


## model
* MF

        class MF(nn.Module):
                def __init__(self, num_factors, num_users, num_items):
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
* DeepFM

        class DeepFM(nn.Module):
            def __init__(self, field_dic, emb_dim, num_factors, mlp_dims, drop_rate=0.1):
                super(DeepFM, self).__init__()

                self.ind_embedding = nn.Embedding(field_dic, emb_dim)
                self.car_embedding = nn.Embedding(field_dic, emb_dim)
                self.reg_embedding = nn.Embedding(field_dic, emb_dim)
                self.calc_embedding = nn.Embedding(field_dic, emb_dim)

                self.fc = nn.Embedding(field_dic, 1)
                self.linear_layer = nn.Linear(1,1)

                input_dim = self.embed_output_dim = num_factors*emb_dim
                self.modules = []
                for dim in mlp_dims:      
                    self.modules.append(nn.Linear(input_dim, dim))
                    self.modules.append(nn.Sigmoid())
                    self.modules.append(nn.Dropout(drop_rate))
                    input_dim = dim
                self.modules.append(nn.Linear(dim,1))
                self.mlp = nn.Sequential(*self.modules)

                self.classify_layer = nn.Linear(1,2)

            def forward(self, ind, car, reg, calc):
                x = torch.cat([ind, car, reg, calc],1).to(device)

                embed_ind = self.ind_embedding(ind)
                embed_car = self.car_embedding(car)
                embed_reg = self.reg_embedding(reg)
                embed_calc = self.calc_embedding(calc)
                embed_x = torch.cat([embed_ind, embed_car, embed_reg, embed_calc],1).to(device)

                square_of_sum = torch.sum(embed_x, 1) ** 2
                sum_of_square = torch.sum(embed_x ** 2, 1)

                inputs = embed_x.view(list(x.size())[0],-1)

                x = self.linear_layer(self.fc(x).sum(1)) + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) + self.mlp(inputs)

                x = self.classify_layer(x)
                x = torch.sigmoid(x)

                return x
    
## data

MovieLens 100K Dataset : https://grouplens.org/datasets/movielens/

Porto Seguro’s Safe Driver Prediction : https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data?select=train.csv
