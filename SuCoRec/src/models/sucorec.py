# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
class SuCoRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SuCoRec, self).__init__(config, dataset)
        self.mode = config['mode']

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.mm_image_weight = config['mm_image_weight']
        self.lambda_js = config['lambda_js']
        self.lambda_ui_enhance = config['lambda_ui_enhance']
        self.lambda_ui = config['lambda_ui']
        self.p_drop = config['p_drop']
        self.top_k = config['top_k']
        self.p_step = config['p_step']
        self.eval_interval = config['eval_interval']
        self.enable_residual = config['enable_residual']

        self.local_sample_p = config['local_sample_p']
        self.p_min = config['p_min']
        self.p_max = config['p_max']
        self.best_metric = -np.inf

        self.n_nodes = self.n_users + self.n_items
        self.cached_user_graph = None
        self.user_user_graph_epoch = -1
        self.knn_update_interval = 10
        self.epoch_count = -1



        self.sub_graph = None
        self.mm_adj = None

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(
            self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k,
                                                                                     int(10 * self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj  # 计算多模态邻接矩阵
                del text_adj
                del image_adj
            os.makedirs(os.path.dirname(mm_adj_file), exist_ok=True)
            torch.save(self.mm_adj, mm_adj_file)



    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)

        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def save(self):
        pass

    def pre_epoch_processing(self):
        degree_len = int(self.edge_values.size(0) * self.local_sample_p)
        degree_idx = torch.multinomial(self.edge_values,
                                       degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices,
                                            torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.sub_graph = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)
        self.epoch_count += 1


    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values



    def forward(self, adj, perturbed=False, flag=False, return_h=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if perturbed:
            adj = self.random_perturb_adj(adj, drop_prob=self.p_drop)
        image_feats, text_feats = F.normalize(image_feats), F.normalize(text_feats)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)

        h = item_embeds
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]

        if self.enable_residual:
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                _weights = F.cosine_similarity(side_embeddings, ego_embeddings, dim=-1)
                side_embeddings = torch.einsum('a,ab->ab', _weights, side_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
        else:
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        top_k = self.top_k
        neighbor_indices, neighbor_weights = self.get_cached_user_user_graph(
            u_g_embeddings.detach(), top_k, self.epoch_count
        )
        uu_agg_embeddings = self.user_user_aggregate(u_g_embeddings, neighbor_indices, neighbor_weights)
        if perturbed and flag:
            sigma = 0.2
            h = h + torch.randn_like(h) * sigma
        if return_h:
            return u_g_embeddings + uu_agg_embeddings * 0.05, i_g_embeddings + h, h
        else:
            return u_g_embeddings + uu_agg_embeddings * 0.05, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def user_item_infonce_loss(self, user_embeds, item_embeds, users, pos_items, temperature=0.2):
        u = user_embeds[users]
        i = item_embeds[pos_items]
        u = F.normalize(u, dim=1)
        i = F.normalize(i, dim=1)
        pos_score = torch.sum(u * i, dim=1)
        pos_score = torch.exp(pos_score / temperature)
        all_items = F.normalize(item_embeds, dim=1)
        all_score = torch.matmul(u, all_items.T) / temperature
        all_score = torch.exp(all_score).sum(dim=1)
        loss = -torch.log(pos_score / all_score)
        return torch.mean(loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        user_emb_n1, item_emb_n1 = self.forward(self.sub_graph, perturbed=True)
        user_emb_n2, item_emb_n2 = self.forward(self.sub_graph, perturbed=True)
        user_enhance_loss = self.InfoNCE(user_emb_n1, user_emb_n2, 0.2)
        item_enhance_loss = self.InfoNCE(item_emb_n1, item_emb_n2, 0.2)
        enhance_g_loss = user_enhance_loss + item_enhance_loss
        ua_clean, ia_clean, h_clean = self.forward(self.sub_graph, perturbed=False, flag=False, return_h=True)
        ua_noisy, ia_noisy, h_noisy = self.forward(self.sub_graph, perturbed=True, flag=True, return_h=True)
        bpr_loss = self.bpr_loss(ua_noisy[users], ia_noisy[pos_items], ia_noisy[neg_items])
        distill_loss_student = self.js_divergence(h_noisy, ia_clean.detach())
        distill_loss_teacher = self.js_divergence(ia_clean, h_noisy.detach())
        distill_loss = distill_loss_student + 0.2 * distill_loss_teacher
        ui_cl_loss = self.user_item_infonce_loss(ua_clean, ia_clean, users, pos_items, temperature=0.2)
        total_loss = bpr_loss + self.lambda_js * distill_loss + enhance_g_loss * self.lambda_ui_enhance + self.lambda_ui * ui_cl_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))  #
        return scores

    def build_user_user_graph(self, user_emb, top_k=20):
        user_emb_norm = F.normalize(user_emb, dim=1)
        sim_matrix = user_emb_norm @ user_emb_norm.t()
        values, indices = torch.topk(sim_matrix, top_k + 1, dim=1)
        arange_idx = torch.arange(user_emb.size(0)).unsqueeze(1).to(user_emb.device)
        mask = indices != arange_idx
        indices = indices[mask].view(user_emb.size(0), top_k)
        values = values[mask].view(user_emb.size(0), top_k)
        weights = F.softmax(values, dim=1)
        return indices, weights

    def user_user_aggregate(self, user_emb, neighbor_indices, neighbor_weights):
        neighbor_emb = user_emb[neighbor_indices]
        agg_user_emb = torch.bmm(neighbor_weights.unsqueeze(1), neighbor_emb)
        agg_user_emb = agg_user_emb.squeeze(1)
        return agg_user_emb
    def random_perturb_adj(self, adj, drop_prob=0.2):
        values = adj._values()
        mask = torch.rand(values.size(), device=values.device) > drop_prob
        new_values = values * mask.float()
        return torch.sparse.FloatTensor(adj._indices(), new_values, adj.shape)

    def js_divergence(self, p, q, eps=1e-8):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        p = torch.clamp(p, min=eps, max=1.0)
        q = torch.clamp(q, min=eps, max=1.0)
        m = 0.5 * (p + q)
        m = torch.clamp(m, min=eps, max=1.0)
        js = 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') +
                    F.kl_div(q.log(), m, reduction='batchmean'))
        return js
    def get_cached_user_user_graph(self, user_embeds, top_k, epoch):
        if (
                self.cached_user_graph is not None
                and (epoch % self.knn_update_interval != 0)
        ):
            return self.cached_user_graph
        neighbor_indices, neighbor_weights = self.build_user_user_graph(user_embeds, top_k)
        self.cached_user_graph = (neighbor_indices, neighbor_weights)
        self.user_user_graph_epoch = epoch
        return self.cached_user_graph
