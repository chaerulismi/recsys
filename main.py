import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import RecSysDataset
from models import Baseline
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, default="vanilla")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--nn-dims", type=str, default="64,32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def main(args):

    train_data = RecSysDataset(
        dataset_name=args.dataset_name, split="train", movielens_type=args.dataset_type
    ).get_dataset()
    test_data = RecSysDataset(
        dataset_name=args.dataset_name, split="test", movielens_type=args.dataset_type
    ).get_dataset()

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    num_users = train_data.num_users
    num_items = train_data.num_items
    num_genres = train_data.num_genres
    nn_dims = [int(x) for x in args.nn_dims.split(",")]
    model = Baseline(
        input_size=args.embedding_dim * 3,
        nn_dims=nn_dims,
        output_size=1,
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        embedding_dim=args.embedding_dim,
    )
    loss_fn = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        model_name="baseline",
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
    )
    trainer.train(eval=True)


if __name__ == "__main__":
    args = get_args()
    main(args)
