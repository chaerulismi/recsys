import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# dataset mapping
dataset_keys = {"ml-32m", "ml-1m", "ml-latest-small"}


def padding_seq(seq, max_len, padding_value=0):
    # truncate the input seq
    seq = seq[-max_len:]
    # initialize the output seq with the padding_value
    out_seq = [padding_value] * max_len
    out_mask = [0] * max_len
    for i, item in enumerate(seq):
        out_seq[i] = item
        out_mask[i] = 1
    return (np.array(out_seq), np.array(out_mask))


def padding_seq_2d(seq_2d, max_seq_len, max_len, padding_value=0):
    seq_2d = seq_2d[-max_seq_len:]
    out_seq = [[padding_value] * max_len for _ in range(max_seq_len)]
    out_seq_mask = [[padding_value] * max_len for _ in range(max_seq_len)]
    for i, seq in enumerate(seq_2d):
        seq, seq_mask = padding_seq(seq, max_len=max_len, padding_value=padding_value)
        out_seq[i] = seq
        out_seq_mask[i] = seq_mask
    return (np.array(out_seq), np.array(out_seq_mask))


class MovieLensDataset(Dataset):

    def __init__(
        self,
        dataset_name,
        type="vanilla",
        split="train",
        augmentation=False,
        max_seq_length=30,
    ):
        super().__init__()
        self.data_dir = os.path.join("data", dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.augmentation = augmentation
        self.max_seq_length = max_seq_length
        self.type = type

        if not os.path.exists(os.path.join(self.data_dir, "vanilla_train.csv")):
            self._preprocess()

        # load dataset
        if split == "train":
            filepath = os.path.join(self.data_dir, f"{type}_train.csv")
        elif split == "test":
            filepath = os.path.join(self.data_dir, f"{type}_test.csv")
        else:
            raise ValueError(f"Expected split ['train', 'test'], got {split}")
        self.dataset = pd.read_csv(filepath)
        # load genre vocab
        vocab_path = os.path.join(self.data_dir, "genre_vocab.txt")
        with open(vocab_path, "r") as f:
            genres = sorted(f.read().splitlines())
        self.genre_vocab = {g: i + 1 for i, g in enumerate(genres)}
        # load stats
        stats_path = os.path.join(self.data_dir, "stats.csv")
        df_stats = pd.read_csv(stats_path)
        self.num_users = df_stats["total_users"].tolist()[0]
        self.num_items = df_stats["total_items"].tolist()[0]
        self.num_genres = df_stats["total_genres"].tolist()[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.type == "sequence":
            # get features
            movie_id_seq = [
                int(x) for x in self.dataset.loc[idx, "movieIdSeq"].split(",")
            ]
            movie_id_seq, movie_id_seq_mask = padding_seq(
                movie_id_seq, max_len=self.max_seq_length
            )
            genre_seq = [
                [self.genre_vocab.get(y, 0) for y in x.split("|")]
                for x in self.dataset.loc[idx, "genreSeq"].split(",")
            ]
            genre_seq, genre_seq_mask = padding_seq_2d(
                genre_seq,
                max_seq_len=self.max_seq_length,
                max_len=len(self.genre_vocab) + 1,
            )
            rating_seq = [
                float(x) for x in self.dataset.loc[idx, "ratingSeq"].split(",")
            ]
            rating_seq, rating_seq_mask = padding_seq(
                rating_seq, max_len=self.max_seq_length
            )

            # target item features
            target_movie_id = int(self.dataset.loc[idx, "targetMovieId"])
            target_genre = [
                self.genre_vocab.get(x, 0)
                for x in self.dataset.loc[idx, "targetGenre"].split("|")
            ]
            target_genre, _ = padding_seq(
                target_genre, max_len=len(self.genre_vocab) + 1
            )

            feats = {
                "movie_id_seq": movie_id_seq,
                "movie_id_seq_mask": movie_id_seq_mask,
                "genre_seq": genre_seq,
                "genre_seq_mask": genre_seq_mask,
                "rating_seq": rating_seq,
                "rating_seq_mask": rating_seq_mask,
                "target_movie_id": target_movie_id,
                "target_genre": target_genre,
            }

            # get label
            label = float(self.dataset.loc[idx, "targetRating"])
            return feats, label
        elif self.type == "vanilla":
            # get features
            movie_id = int(self.dataset.loc[idx, "movieId"])
            user_id = int(self.dataset.loc[idx, "userId"])
            genre = [
                self.genre_vocab.get(x, 0)
                for x in self.dataset.loc[idx, "genre"].split("|")
            ]
            genre, genre_mask = padding_seq(genre, max_len=len(self.genre_vocab) + 1)
            feats = {
                "user_id": user_id,
                "movie_id": movie_id,
                "genre": genre,
                "genre_mask": genre_mask,
            }
            # get label
            label = float(self.dataset.loc[idx, "rating"])
            return feats, label
        else:
            raise ValueError(f"Invalid type: {type}!")

    def _preprocess(self):
        print("-- Preprocessing the original MovieLens dataset.")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        movies_path = os.path.join(self.data_dir, "movies.csv")
        genre_path = os.path.join(self.data_dir, "genre_vocab.txt")
        stats_path = os.path.join(self.data_dir, "stats.csv")

        if self.dataset_name == "ml-1m":
            # reformat the .dat into .csv
            ratings = pd.read_csv(
                os.path.join(self.data_dir, "ratings.dat"),
                sep="::",
                header=None,
                names=["userId", "movieId", "rating", "timestamp"],
            )
            ratings.to_csv(ratings_path, index=False)

            movies = pd.read_csv(
                os.path.join(self.data_dir, "movies.dat"),
                sep="::",
                header=None,
                names=["movieId", "title", "genres"],
                encoding="ISO-8859-1",
            )
            movies.to_csv(movies_path, index=False)

        if not os.path.exists(ratings_path):
            raise ValueError("ratings.csv not found!")
        if not os.path.exists(movies_path):
            raise ValueError("movies.csv not found!")

        # load ratings
        ratings = pd.read_csv(ratings_path)
        ratings["month"] = ratings["timestamp"].map(
            lambda ts: datetime.datetime.fromtimestamp(ts).month
        )
        ratings["day"] = ratings["timestamp"].map(
            lambda ts: datetime.datetime.fromtimestamp(ts).day
        )
        ratings["hour"] = ratings["timestamp"].map(
            lambda ts: datetime.datetime.fromtimestamp(ts).hour
        )

        # load movies
        movies = pd.read_csv(movies_path)
        genre_vocab = set()
        for genre in movies["genres"]:
            for g in genre.split("|"):
                genre_vocab.add(g)
        print(f"-- Genre vocab size: {len(genre_vocab)}")
        with open(genre_path, "w") as f:
            f.write("\n".join(sorted(genre_vocab)))
        # get unique items
        print(f"-- Total items: {max(movies['movieId'].values.tolist())}")

        df = pd.merge(left=ratings, right=movies, how="left", on="movieId").reset_index(
            drop=True
        )

        # get unique users
        users = df["userId"].unique()
        print(f"-- Total users: {len(users)}")

        df_stats = pd.DataFrame(
            {
                "total_users": [len(users)],
                "total_items": [max(movies["movieId"].values.tolist())],
                "total_genres": [len(genre_vocab)],
            }
        )
        df_stats.to_csv(stats_path, index=False)

        # convert ratings to binary
        df["rating"] = df["rating"].map(lambda x: 1 if x >= 3 else 0)

        # generate datasets
        test_user_ids = []
        test_feat_genres = []
        test_feat_movie_ids = []
        test_feat_ratings = []
        test_feat_months = []
        test_feat_days = []
        test_feat_hours = []
        test_label_genres = []
        test_label_movie_ids = []
        test_label_ratings = []
        test_label_months = []
        test_label_days = []
        test_label_hours = []

        train_user_ids = []
        train_feat_genres = []
        train_feat_movie_ids = []
        train_feat_ratings = []
        train_feat_months = []
        train_feat_days = []
        train_feat_hours = []
        train_label_genres = []
        train_label_movie_ids = []
        train_label_ratings = []
        train_label_months = []
        train_label_days = []
        train_label_hours = []

        # vanilla dataset
        train_vanilla_dfs = []
        for uid in tqdm(users):
            tmp = (
                df[df["userId"] == uid].sort_values("timestamp").reset_index(drop=True)
            )
            if len(tmp) < 20:
                continue
            tmp_movies = tmp["movieId"].astype(str).tolist()
            tmp_ratings = tmp["rating"].astype(str).tolist()
            tmp_genres = tmp["genres"].astype(str).tolist()
            tmp_months = tmp["month"].astype(str).tolist()
            tmp_days = tmp["day"].astype(str).tolist()
            tmp_hours = tmp["hour"].astype(str).tolist()

            # generate test dataset
            test_size = int(len(tmp_movies) * 0.2)
            if test_size > 20:
                test_size = 20

            for _ in range(test_size):
                label_genre = tmp_genres.pop()
                label_movid_id = tmp_movies.pop()
                label_rating = tmp_ratings.pop()
                label_month = tmp_months.pop()
                label_day = tmp_days.pop()
                label_hour = tmp_hours.pop()

                feat_genre = ",".join(tmp_genres)
                feat_movie_id = ",".join(tmp_movies)
                feat_rating = ",".join(tmp_ratings)
                feat_month = ",".join(tmp_months)
                feat_day = ",".join(tmp_days)
                feat_hour = ",".join(tmp_hours)

                test_user_ids.append(str(uid))

                test_feat_genres.append(feat_genre)
                test_feat_movie_ids.append(feat_movie_id)
                test_feat_ratings.append(feat_rating)
                test_feat_months.append(feat_month)
                test_feat_days.append(feat_day)
                test_feat_hours.append(feat_hour)

                test_label_genres.append(label_genre)
                test_label_movie_ids.append(label_movid_id)
                test_label_ratings.append(label_rating)
                test_label_months.append(label_month)
                test_label_days.append(label_day)
                test_label_hours.append(label_hour)

            # generate train vanilla dataset
            vanilla_user_ids = [uid] * len(tmp_ratings)
            tmp_vanilla = pd.DataFrame(
                {
                    "userId": vanilla_user_ids,
                    "movieId": tmp_movies,
                    "rating": tmp_ratings,
                    "genre": tmp_genres,
                    "month": tmp_months,
                    "day": tmp_days,
                    "hour": tmp_hours,
                }
            )
            train_vanilla_dfs.append(tmp_vanilla)

            # generate train dataset for seq
            train_size = int(len(tmp_ratings) * 0.5)
            if train_size > 200:
                train_size = 200

            for _ in range(train_size):
                label_genre = tmp_genres.pop()
                label_movid_id = tmp_movies.pop()
                label_rating = tmp_ratings.pop()
                label_month = tmp_months.pop()
                label_day = tmp_days.pop()
                label_hour = tmp_hours.pop()

                feat_genre = ",".join(tmp_genres)
                feat_movie_id = ",".join(tmp_movies)
                feat_rating = ",".join(tmp_ratings)
                feat_month = ",".join(tmp_months)
                feat_day = ",".join(tmp_days)
                feat_hour = ",".join(tmp_hours)

                train_user_ids.append(str(uid))

                train_feat_genres.append(feat_genre)
                train_feat_movie_ids.append(feat_movie_id)
                train_feat_ratings.append(feat_rating)
                train_feat_months.append(feat_month)
                train_feat_days.append(feat_day)
                train_feat_hours.append(feat_hour)

                train_label_genres.append(label_genre)
                train_label_movie_ids.append(label_movid_id)
                train_label_ratings.append(label_rating)
                train_label_months.append(label_month)
                train_label_days.append(label_day)
                train_label_hours.append(label_hour)

        # constructing train and test vanilla dataset
        train_vanilla_df = pd.concat(train_vanilla_dfs)
        print("-- Train vanilla shape: {}".format(train_vanilla_df.shape))
        train_vanilla_df.to_csv(
            os.path.join(self.data_dir, "vanilla_train.csv"), index=False
        )

        test_vanilla_df = pd.DataFrame(
            {
                "userId": test_user_ids,
                "movieId": test_label_movie_ids,
                "genre": test_label_genres,
                "rating": test_label_ratings,
                "month": test_label_months,
                "day": test_label_days,
                "hour": test_label_hours,
            }
        )
        print("-- Test vanilla shape: {}".format(test_vanilla_df.shape))
        test_vanilla_df.to_csv(
            os.path.join(self.data_dir, "vanilla_test.csv"), index=False
        )

        # construct training and test dataset
        train_df = pd.DataFrame(
            {
                "userId": train_user_ids,
                "movieIdSeq": train_feat_movie_ids,
                "genreSeq": train_feat_genres,
                "ratingSeq": train_feat_ratings,
                "monthSeq": train_feat_months,
                "daySeq": train_feat_days,
                "hourSeq": train_feat_hours,
                "targetMovieId": train_label_movie_ids,
                "targetGenre": train_label_genres,
                "targetRating": train_label_ratings,
                "targetMonth": train_label_months,
                "targetDay": train_label_days,
                "targetHour": train_label_hours,
            }
        )
        print("-- Train sequence shape: {}".format(train_df.shape))
        train_df.to_csv(os.path.join(self.data_dir, "sequence_train.csv"), index=False)

        test_df = pd.DataFrame(
            {
                "userId": test_user_ids,
                "movieIdSeq": test_feat_movie_ids,
                "genreSeq": test_feat_genres,
                "ratingSeq": test_feat_ratings,
                "monthSeq": test_feat_months,
                "daySeq": test_feat_days,
                "hourSeq": test_feat_hours,
                "targetMovieId": test_label_movie_ids,
                "targetGenre": test_label_genres,
                "targetRating": test_label_ratings,
                "targetMonth": test_label_months,
                "targetDay": test_label_days,
                "targetHour": test_label_hours,
            }
        )
        print("-- Test sequence shape: {}".format(test_df.shape))
        test_df.to_csv(os.path.join(self.data_dir, "sequence_test.csv"), index=False)
        print("-- Preprocessing is completed...")


class RecSysDataset(object):

    def __init__(
        self,
        dataset_name,
        split="train",
        movielens_type="vanilla",
        augmentation=False,
        max_seq_length=200,
    ):
        self.dataset = None
        if dataset_name not in dataset_keys:
            raise ValueError(
                f"Invalid dataset-name: {dataset_name}.\n"
                f"Should be one of the following: {dataset_keys}"
            )
        if dataset_name in ("ml-32m", "ml-latest-small", "ml-1m"):
            self.dataset = MovieLensDataset(
                dataset_name=dataset_name,
                split=split,
                type=movielens_type,
                augmentation=augmentation,
                max_seq_length=max_seq_length,
            )

    def get_dataset(self):
        return self.dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    movielens_dataset = RecSysDataset(
        dataset_name="ml-latest-small", movielens_type="vanilla"
    ).get_dataset()
    dataloader = DataLoader(movielens_dataset, batch_size=4)
    features, labels = next(iter(dataloader))
    print(features["genre"])
    print(features["genre_mask"])
    print(labels)
