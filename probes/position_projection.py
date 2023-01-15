import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm


class PositionProjection(nn.Module):
    def __init__(self, input_dimension, rank):
        super(PositionProjection, self).__init__()
        self.input_dimension = input_dimension
        self.rank = rank
        self.proj = nn.Parameter(data=torch.zeros(self.input_dimension, rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, pe):
        transformed = torch.matmul(pe, self.proj)
        pos_size, rank = transformed.size()
        transformed = transformed.unsqueeze(1)
        transformed = transformed.expand(-1, pos_size, -1)
        transposed = transformed.transpose(0, 1)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


def relative_position_probe(x, y, patience=30,
                            props=(0.7, 0.1, 0.2),
                            epochs=500,
                            lr=1e-2,
                            rank=16):
    logger = logging.getLogger()
    dis_y = []
    for y1 in y:
        to_add = []
        for y2 in y:
            to_add.append(abs(y1 - y2))
        dis_y.append(to_add)
    dis_y = np.array(dis_y)

    all_x = torch.tensor(x)
    train_samples = int(props[0] * len(y))
    val_samples = int(props[1] * len(y))
    train_dis_y = torch.tensor(dis_y[0:train_samples, 0:train_samples])
    train_x = torch.tensor(x[0:train_samples])

    model = PositionProjection(x.shape[1], rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_eval_loss = float('inf')
    patience_count = 0
    best_check_point = None
    for e in tqdm(range(epochs), desc='Epochs'):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        number_elements = (train_dis_y.shape[0] ** 2 - train_dis_y.shape[0]) / 2.
        loss = torch.sum(
            torch.abs(torch.triu(pred, diagonal=1) - torch.triu(train_dis_y, diagonal=1))) / number_elements
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_all = model(all_x)
            val_x = torch.tensor(dis_y[train_samples:train_samples + val_samples, :train_samples + val_samples])
            pred_val = pred_all[train_samples:train_samples + val_samples, :train_samples + val_samples]
            val_mae = torch.mean(torch.abs(val_x - pred_val))
            # logger.info(f'Epoch {e} loss train {loss.item():.4f}, loss eval {val_mae.item():.4f}')
            if val_mae < best_eval_loss:
                best_check_point = model.state_dict()
                patience_count = 0
                best_eval_loss = val_mae
            else:
                patience_count += 1
            if patience_count == patience:
                logger.info('Stopping training loop (out of patience).')
                break

    model = PositionProjection(x.shape[1], rank)
    model.load_state_dict(best_check_point)
    model.eval()

    pred_all = model(all_x)
    pred_test = pred_all[train_samples + val_samples:, :]
    test_x = torch.tensor(dis_y[train_samples + val_samples:, :])
    diffs = torch.abs(test_x - pred_test)
    test_mse = torch.mean(diffs)
    logger.info(f'Rank {rank} Test MSE: {test_mse}')
    return test_mse.item()


def relative_position_probe_analysis(x, y, patience=30,
                                     epochs=500,
                                     props=(0.7, 0.1, 0.2),
                                     lr=1e-2):
    test_mse = []
    ranks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for r in ranks:
        test_mse.append(relative_position_probe(x, y, rank=r, epochs=epochs, lr=lr, props=props, patience=patience))
    to_save = pd.DataFrame.from_dict({'Rank': ranks, 'MSE': test_mse})
    to_save.to_csv('relative_position_probe.csv', index=False)


def spearman_coefficient(test, pred):
    coefs = []
    for i in range(test.shape[0]):
        coefs.append(spearmanr(test[i], pred[i]).correlation)
    return np.mean(coefs)
