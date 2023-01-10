import logging
import math
import random

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from plotnine import *
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModel

from probes.position_projection import relative_position_probe

logger = logging.getLogger()


def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def load_pe(cfg):
    if cfg['pe_model'] == 'sinusoid':
        return get_sinusoid(cfg['params']['length_pe'], cfg['params']['dimension'])
    elif cfg['pe_model'] == 'roberta-base' or \
            cfg['pe_model'] == 'microsoft/codebert-base' or \
            cfg['pe_model'] == 'huggingface/CodeBERTa-small-v1' or \
            cfg['pe_model'] == 'microsoft/graphcodebert-base':
        pes = AutoModel.from_pretrained(cfg['pe_model']).embeddings.position_embeddings.weight.data[2:]
        return pes[0:cfg['params']['length_pe']]
    elif cfg['pe_model'] == 'random':
        return torch.randn(cfg['params']['length_pe'], cfg['params']['dimension'])
    else:
        raise ValueError('Only sinusoid, roberta-base and microsoft/codebert-base are supported')


def mse(model, x, y):
    return (np.abs((np.round(model.predict(x)) - y))).mean()


def absolute_position_analysis(pe):
    y = np.random.permutation(pe.shape[0])
    x = pe[y]
    model = LinearRegression()
    model.fit(x, y)
    score = cross_val_score(model, x, y, cv=5, scoring=mse).mean()
    logger.info(f'MSE absolute position analysis {score:.2f}')


def relative_position_analysis(pe):
    x = pe
    r = np.random.permutation(x.shape[0] ** 2)
    y1 = r[:x.shape[0]] // x.shape[0]
    y2 = r[:x.shape[0]] % x.shape[0]
    x1 = x[y1]
    x2 = x[y2]
    x = np.hstack((x1, x2))
    y = y2 > y1
    model = LogisticRegression(max_iter=1000)
    score = cross_val_score(model, x, y, cv=5).mean()
    logger.info(f'Accuracy for relative position analysis {score:.4f}')


def position_wise_cosine_similarity(pe):
    cosine_data = {'position1': [], 'position2': [], 'cosine_sim': []}
    for i in range(pe.shape[0]):
        for j in range(pe.shape[0]):
            cosine_data['cosine_sim'].append(1. - cosine(pe[i], pe[j]))
            cosine_data['position1'].append(i)
            cosine_data['position2'].append(j)
    to_plot = pd.DataFrame.from_dict(cosine_data)
    heatmap_cosine = (
            ggplot(mapping=aes("position1", "position2", fill="cosine_sim"),
                   data=to_plot)
            + geom_tile()
            + scale_fill_distiller()
            + theme_minimal()
            + labs(title="", x="", y="", fill="cosine\n")
            + theme(legend_title=element_text(size=12),
                    legend_title_align='center')
    )
    heatmap_cosine.save(f"heatmap_cosine.pdf", dpi=600)


def eigenvalues_analysis(pe):
    u, s, vh = np.linalg.svd(pe, full_matrices=True)
    s = s / np.sum(s)
    s = np.sort(s)[::-1]
    for i in range(0, len(s) + 50, 50):
        top_sum = np.sum(s[0:i])
        logger.info(f'Top {i}: {top_sum}')

    data_to_save = {'topx': [], 'accumulation': []}
    for i in range(0, len(s)):
        top_sum = np.sum(s[0:i])
        data_to_save['topx'].append(i)
        data_to_save['accumulation'].append(top_sum)
    to_save = pd.DataFrame.from_dict(data_to_save)
    to_save.to_csv('eigen_accumulation.csv', index=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg['params']['seed'])
    pe = load_pe(cfg).numpy()
    if cfg['analysis'] == 'absolute_position':
        absolute_position_analysis(pe)
    if cfg['analysis'] == 'relative_position':
        relative_position_analysis(pe)
    if cfg['analysis'] == 'position_wise_cosine_similarity':
        position_wise_cosine_similarity(pe)
    if cfg['analysis'] == 'eigenvalues_analysis':
        eigenvalues_analysis(pe)
    if cfg['analysis'] == 'relative_position_probe':
        relative_position_probe(pe)


if __name__ == '__main__':
    main()
