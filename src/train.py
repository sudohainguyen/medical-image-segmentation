# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np
import click
from glob import glob
from tqdm import tqdm

import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import KFold

from .models import FDSE
from .dataset import DatasetB
from .utils import Params
from .helpers import TrainingModule


def _get_fnames(datadir):
    fnames = glob(os.path.join(datadir, 'images'))
    fnames = [os.path.splitext(os.path.basename(path))[0] for path in fnames]
    fnames = np.array(fnames)
    return fnames


@click.command()
@click.option('--datadir', default='.')
@click.option('--n_classes', default=1)
@click.option('--batch_size', default=16)
@click.option('--lr', default=1e-5)
@click.option('--epochs', default=50)
@click.option('--n_folds', default=5)
@click.option('--mid_channels', default=512)
def train(
    datadir,
    n_classes=1,
    batch_size=16,
    lr=1e-4,
    epochs=50,
    n_folds=5,
    mid_channels=512,
    shuffle=True,
    num_workers=1,
    seed=100
):
    params = Params(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        n_folds=n_folds,
        mid_channels=mid_channels
    )

    np.random.seed(params.seed)

    kfold = KFold(n_splits=params.n_folds)
    fnames = _get_fnames(datadir)
    fnames = np.random.shuffle(fnames)

    with mlflow.start_run():
        for fold, (train_fns, val_fns) in enumerate(kfold.split(fnames)):
            model = FDSE(
                n_classes=1,
                mid_channels=params.mid_channels
            )
            criterion = nn.BCELoss()
            optimizer = Adam(model.parameters(), lr=params.lr)
            # optimizer = SGD(model.parameters(), 
            #                 lr=params.lr, momentum=params.momentum)

            trainer = Trainer(n_classes, criterion, optimizer)
            evaluator = Evaluator(n_classes, criterion)

            train_ds = DatasetB(train_fns)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers)
            
            val_ds = DatasetB(val_fns)
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)

            for e in range(epochs):
                # model.train()
                tbar = tqdm(train_loader)
                train_loss = 0
                for i, batch in enumerate(tbar):
                    loss = trainer.train(model, batch)
                    train_loss += loss.item()
                    cur_loss = train_loss / (i + 1)
                    cur_score = trainer.get_scores()
                    tbar.set_description(
                        "Train loss: %.3f; Train Dice: %.3f"
                        % (
                            cur_loss,
                            np.mean(cur_score['train_dice'])
                        )
                    )
                    
                train_scores = trainer.metrics.get_scores()
                trainer.reset_metrics()
                mlflow.log_metrics(train_scores)

                # model.eval()
                tbar = tqdm(val_loader)
                for i, batch in enumerate(tbar):
                    evaluator.update(model, batch)
                    cur_scores = evaluator.get_scores()
                    
                    tbar.set_description(
                        "val_dice: %.3f" % (cur_scores['val_dice'])
                    )

                val_scores = evaluator.get_scores(phase='val')
                # mlflow.log_metrics(val_scores)
            torch.save(model.state_dict(), f'model_fold_{fold}.pth')
            # mlflow.pytorch.save_model(model, f'model_fold_{fold}.pth')
        # end of experiments
        # mlflow.log_params(params.__dict__)


if __name__ == "__main__":
    train()
