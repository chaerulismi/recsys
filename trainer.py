from datetime import datetime

import pandas as pd
import torch
import torch.cuda
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm


class Trainer(object):

    def __init__(
        self,
        train_dataloader,
        test_dataloader,
        dataset_name,
        dataset_type,
        model_name,
        model,
        loss_fn,
        optimizer,
        epochs=5,
    ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = {"BinaryAUCROC": BinaryAUROC()}
        print(f"Using {self.device} device!")
        # log the training progress
        curr_ts = datetime.now()
        training_id = curr_ts.strftime("%Y%m%d%H%M%S")
        self.log_filename = (
            f"log/{dataset_name}_{dataset_type}_{model_name}_{training_id}.csv"
        )
        self.logs = []

    def train(self, eval=False):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch}") as pbar:
                for X, y in self.train_dataloader:
                    # X, y = X.to(self.device), y.to(self.device)

                    # compute prediction error
                    pred = self.model(X)
                    pred = pred.squeeze(1)
                    y = y.float()
                    loss = self.loss_fn(pred, y)

                    # backpropagation
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update()
                if eval:
                    test_loss, auc = self.evaluate()
                    self.logs.append([epoch, test_loss, auc])
        # write the training progress into csv file
        log_df = pd.DataFrame(
            {
                "epoch": [x[0] for x in self.logs],
                "loss": [x[1] for x in self.logs],
                "auc": [x[2] for x in self.logs],
            }
        )
        log_df.to_csv(self.log_filename, index=False)

    def evaluate(self):
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                # X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                pred = pred.squeeze(1)
                y = y.float()
                test_loss += self.loss_fn(pred, y).item()
                self.metrics["BinaryAUCROC"].update(pred, y)
        test_loss /= num_batches
        print("Test Error:")
        print(f"- BinaryAUCROC: {self.metrics['BinaryAUCROC'].compute():.4f}")
        print(f"- Loss: {test_loss:.8f}")
        return (test_loss, self.metrics["BinaryAUCROC"].compute().numpy())
