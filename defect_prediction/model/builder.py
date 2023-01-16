import math

import torch.nn as nn
import torch
from transformers import AutoModel


def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerClassifier(nn.Module):
    def __init__(self, embedding, n_class, nhead=8, dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        d_model = embedding.dim
        self.embedding = embedding
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(layer, 1,
                                             nn.LayerNorm(d_model))
        self.d_model = d_model
        self.fc = nn.Linear(d_model, n_class)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask=None, label=None):
        emb = self.embedding(x)

        # (N, S, E) -> (S, N, E)
        emb = emb.transpose(0, 1)

        padding_mask = (mask == 0)
        hidden = self.encoder(emb, src_key_padding_mask=padding_mask)

        # (S, N, E) -> (N, S, E)
        hidden = hidden.transpose(0, 1)

        # last or first one?
        logit = self.fc(hidden[:, 0, :])
        if label is not None:
            loss = self.criterion(logit, label)
            return loss, logit
        else:
            return torch.softmax(logit, dim=-1)


class PositionEmbedding(nn.Module):
    def __init__(self, n_emb, dim=256, dropout=0.1, max_len=128):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.word_emb = nn.Embedding(n_emb, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        max_len = x.size(-1)
        ids = torch.arange(max_len, dtype=torch.long, device=x.device)
        ids = ids.unsqueeze(0).expand(x.size())
        word_emb = self.word_emb(x)
        pos_emb = self.pos_emb(ids)
        emb = word_emb + pos_emb
        return self.dropout(emb)


def build_model(vocab_size, num_labels, d_model=768, max_len=512, pe='random'):
    if pe == 'random':
        embedding = PositionEmbedding(vocab_size, max_len=max_len, dim=d_model)
    elif pe == 'sinusoid':
        embedding = PositionEmbedding(vocab_size, max_len=max_len, dim=d_model)
        sinusoid_pe = get_sinusoid(max_len, d_model)
        embedding.pos_emb.weight.data = sinusoid_pe
        embedding.pos_emb.weight.requires_grad = False
    elif pe == 'gpt2':
        embedding = PositionEmbedding(vocab_size, max_len=max_len, dim=d_model)
        gpt2_pe = AutoModel.from_pretrained(pe).wpe.weight.data[0:max_len]
        embedding.pos_emb.weight.data = gpt2_pe
        embedding.pos_emb.weight.requires_grad = False
    elif pe == 'microsoft/codebert-base' or pe == 'roberta-base' or pe == 'huggingface/CodeBERTa-small-v1':
        embedding = PositionEmbedding(vocab_size, max_len=max_len, dim=d_model)
        pt_pe = AutoModel.from_pretrained(pe).embeddings.position_embeddings.weight.data[2:]
        embedding.pos_emb.weight.data = pt_pe
        embedding.pos_emb.weight.requires_grad = False
    elif pe == 'bow':
        embedding = PositionEmbedding(vocab_size, max_len=max_len, dim=d_model)
        embedding.pos_emb.weight.data = torch.zeros(max_len, d_model)
        embedding.pos_emb.weight.requires_grad = False
    else:
        raise ValueError(f'PE {pe} not supported')
    model = TransformerClassifier(embedding, num_labels)
    return model


def build_model_checkpoint(vocab_size, num_labels, checkpoint, d_model=768, max_len=512, pe='random'):
    model = build_model(vocab_size, num_labels, d_model, max_len, pe)
    model.load_state_dict(torch.load(checkpoint))
    return model
