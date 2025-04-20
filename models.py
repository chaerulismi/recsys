import torch
from torch import nn


class Baseline(nn.Module):

    def __init__(self,
                 input_size,
                 nn_dims,
                 output_size,
                 num_users,
                 num_items,
                 embedding_dim=32,
                 num_genres=20):
        super().__init__()
        layers = []
        # create user and item embedding layer
        self.user_emb = nn.Embedding(num_users + 2, embedding_dim)
        self.item_emb = nn.Embedding(num_items + 2, embedding_dim)
        self.genre_emb = nn.Embedding(num_genres + 2, embedding_dim)
        # create the first layer
        layers.append(nn.Linear(input_size, nn_dims[0]))
        layers.append(nn.ReLU())
        # create the subsequent hidden layers
        for i in range(len(nn_dims) - 1):
            layers.append(nn.Linear(nn_dims[i], nn_dims[i + 1]))
            layers.append(nn.ReLU())
        # create the final output layer
        layers.append(nn.Linear(nn_dims[-1], output_size))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_dict):
        user_ids = input_dict['user_id']  # (bs,)
        item_ids = input_dict['movie_id']  # (bs,)
        genre = input_dict['genre']  # (bs, num_genres)
        genre_mask = input_dict['genre_mask']  # (bs, num_genres)
        user_embeddings = self.user_emb(user_ids)  # (bs, emb_dim)
        item_embeddings = self.item_emb(item_ids)  # (bs, emb_dim)
        genre_embeddings = self.genre_emb(genre)  # (bs, num_genres, emb_dim)
        mask_float = genre_mask.float()
        mask_expanded = mask_float.unsqueeze(-1)  # (bs, num_genres, 1)
        masked_sum = (genre_embeddings * mask_expanded).sum(
            dim=1)  # (bs, emb_dim)
        valid_counts = mask_float.sum(dim=1, keepdim=True)  # (bs, 1)
        valid_counts = torch.clamp(valid_counts, min=1e-9)
        pooled_mean = masked_sum / valid_counts  # (bs, emb_dim)
        x = torch.concat([user_embeddings, item_embeddings, pooled_mean],
                         dim=1)  # (bs, emb_dim*3)
        return self.mlp(x)
