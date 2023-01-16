import logging
import os

import hydra
from omegaconf import DictConfig
from tokenizers.implementations import ByteLevelBPETokenizer

from data import read_datasets


def batch_iterator(dataset, batch_size=100):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["func"]


BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


def train_tokenizer(train_set, cfg):
    # model
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(batch_iterator(train_set),
                                  vocab_size=cfg["vocab_size"],
                                  min_frequency=2,
                                  special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN])

    os.makedirs(cfg['tokenizer_checkpoint'], exist_ok=True)
    tokenizer.save_model(cfg['tokenizer_checkpoint'])


@hydra.main(version_base=None, config_path=".", config_name="config_tokenizer")
def main(cfg: DictConfig):
    logger = logging.getLogger()

    train_set, _, _ = read_datasets(cfg)
    logger.info(f'Training samples {len(train_set)}')
    train_tokenizer(train_set, cfg)


if __name__ == '__main__':
    main()
