import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

# dataset mapping
dataset_map = {
    'movielens-32m': 'ml-32m',
    'movielens-latest-small': 'ml-latest-small'
}


def padding_seq(seq, max_len, padding_value=0):
    # truncate the input seq
    seq = seq[-max_len:]
    # initialize the output seq with the padding_value
    out_seq = [padding_value] * max_len
    out_mask = [0] * max_len
    for i, item in enumerate(seq):
        out_seq[i] = item
        out_mask[i] = 1
    return (out_seq, out_mask)


def padding_seq_2d(seq_2d, max_seq_len, max_len, padding_value=0):
    seq_2d = seq_2d[-max_seq_len:]
    out_seq = [[padding_value] * max_len for _ in range(max_seq_len)]
    out_seq_mask = [[padding_value] * max_len for _ in range(max_seq_len)]
    for i, seq in enumerate(seq_2d):
        seq, seq_mask = padding_seq(seq,
                                    max_len=max_len,
                                    padding_value=padding_value)
        out_seq[i] = seq
        out_seq_mask[i] = seq_mask
    return (out_seq, out_seq_mask)


class MovieLensDataset(Dataset):

    def __init__(self,
                 data_dir,
                 type='vanilla',
                 split='train',
                 augmentation=False,
                 max_seq_length=200):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.augmentation = augmentation
        self.max_seq_length = max_seq_length
        self.type = type
        # load dataset
        if split == 'train':
            filepath = os.path.join(data_dir, f'{type}_train.csv')
        elif split == 'test':
            filepath = os.path.join(data_dir, f'{type}_test.csv')
        else:
            raise ValueError(f"Expected split ['train', 'test'], got {split}")
        self.dataset = pd.read_csv(filepath)
        # load genre vocab
        vocab_path = os.path.join(data_dir, 'genre_vocab.txt')
        with open(vocab_path, 'r') as f:
            genres = sorted(f.read().splitlines())
        self.genre_vocab = {g: i + 1 for i, g in enumerate(genres)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.type == 'sequence':
            # get features
            movie_id_seq = [
                int(x) for x in self.dataset.loc[idx, 'movieIdSeq'].split(',')
            ]
            movie_id_seq, movie_id_seq_mask = padding_seq(
                movie_id_seq, max_len=self.max_seq_length)
            genre_seq = [[self.genre_vocab.get(y, 0)
                          for y in x.split('|')]
                         for x in self.dataset.loc[idx, 'genreSeq'].split(',')]
            genre_seq, genre_seq_mask = padding_seq_2d(
                genre_seq,
                max_seq_len=self.max_seq_length,
                max_len=len(self.genre_vocab) + 1)
            rating_seq = [
                float(x) for x in self.dataset.loc[idx, 'ratingSeq'].split(',')
            ]
            rating_seq, rating_seq_mask = padding_seq(
                rating_seq, max_len=self.max_seq_length)

            # target item features
            target_movie_id = int(self.dataset.loc[idx, 'targetMovieId'])
            target_genre, _ = padding_seq(str(
                self.dataset.loc[idx, 'targetGenre']).split('|'),
                                          max_len=self.genre_vocab + 1)

            feats = {
                'movie_id_seq': movie_id_seq,
                'movie_id_seq_mask': movie_id_seq_mask,
                'genre_seq': genre_seq,
                'genre_seq_mask': genre_seq_mask,
                'rating_seq': rating_seq,
                'rating_seq_mask': rating_seq_mask,
                'target_movie_id': target_movie_id,
                'target_genre': target_genre
            }

            # get label
            label = float(self.dataset.loc[idx, 'targetRating'])
            return feats, label
        elif self.type == 'vanilla':
            # get features
            movie_id = int(self.dataset.loc[idx, 'movieId'])
            user_id = int(self.dataset.loc[idx, 'userId'])
            genre, genre_mask = padding_seq(
                self.dataset.loc[idx, 'genre'].split('|'),
                max_len=len(self.genre_vocab) + 1)
            feats = {
                'user_id': user_id,
                'movie_id': movie_id,
                'genre': genre,
                'genre_mask': genre_mask
            }
            # get label
            label = float(self.dataset.loc[idx, 'rating'])
            return feats, label
        else:
            raise ValueError(f"Invalid type: {type}!")


class RecSysDataset(object):

    def __init__(self,
                 dataset_name,
                 split='train',
                 augmentation=False,
                 max_seq_length=200):
        if dataset_name not in dataset_map:
            raise ValueError(
                f"Invalid dataset-name: {dataset_name}.\n"
                "Should be one of the following: {dataset_map.keys()}")
        if dataset_name in ('movielens-32m'):
            data_dir = f"data/{dataset_map[self.dataset_name]}"
            return MovieLensDataset(data_dir=data_dir,
                                    split=split,
                                    augmentation=augmentation,
                                    max_seq_length=max_seq_length)


if __name__ == '__main__':
    movielens_dataset = RecSysDataset(dataset_name='movielens-latest-small')
