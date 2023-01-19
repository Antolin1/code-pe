import math

import torch
import torch.nn as nn
from transformers import AutoModel


def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerEncoder(nn.Module):
    def __init__(self, embedding, nhead=8, dim_feedforward=512,
                 dropout=0.1, n_layers=1):
        super().__init__()
        d_model = embedding.dim
        self.embedding = embedding
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             nn.LayerNorm(d_model))
        self.d_model = d_model

    def forward(self, x, mask=None):
        emb = self.embedding(x)

        # (N, S, E) -> (S, N, E)
        emb = emb.transpose(0, 1)

        padding_mask = (mask == 0)
        hidden = self.encoder(emb, src_key_padding_mask=padding_mask)

        # (S, N, E) -> (N, S, E)
        hidden = hidden.transpose(0, 1)

        # last or first one?
        return hidden[:, 0, :]


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


class DualEconderModel(nn.Module):
    def __init__(self, code_encoder, nl_encoder):
        super().__init__()
        self.code_encoder = code_encoder
        self.nl_encoder = nl_encoder

    def forward(self, input_ids_code, inputs_ids_nl,
                attention_mask_code, attention_mask_nl):
        emb_code = self.code_encoder(input_ids_code, attention_mask_code)
        emb_nl = self.nl_encoder(inputs_ids_nl, attention_mask_nl)
        return emb_code, emb_nl


def build_model(vocab_size_code, vocab_size_nl, cfg):
    pe = cfg["pe_model"]
    max_len = cfg["training_params"]["block_size_code"]
    d_model = cfg["model_params"]["d_model_code"]
    if pe == 'random':
        embedding_code = PositionEmbedding(vocab_size_code, max_len=max_len, dim=d_model)
    elif pe == 'sinusoid':
        embedding_code = PositionEmbedding(vocab_size_code, max_len=max_len, dim=d_model)
        sinusoid_pe = get_sinusoid(max_len, d_model)
        embedding_code.pos_emb.weight.data = sinusoid_pe
        embedding_code.pos_emb.weight.requires_grad = False
    elif pe == 'gpt2':
        embedding_code = PositionEmbedding(vocab_size_code, max_len=max_len, dim=d_model)
        gpt2_pe = AutoModel.from_pretrained(pe).wpe.weight.data[0:max_len]
        embedding_code.pos_emb.weight.data = gpt2_pe
        embedding_code.pos_emb.weight.requires_grad = False
    elif pe == 'microsoft/codebert-base' or pe == 'roberta-base' or pe == 'huggingface/CodeBERTa-small-v1':
        embedding_code = PositionEmbedding(vocab_size_code, max_len=max_len, dim=d_model)
        pt_pe = AutoModel.from_pretrained(pe).embeddings.position_embeddings.weight.data[2:]
        embedding_code.pos_emb.weight.data = pt_pe
        embedding_code.pos_emb.weight.requires_grad = False
    elif pe == 'bow':
        embedding_code = PositionEmbedding(vocab_size_code, max_len=max_len, dim=d_model)
        embedding_code.pos_emb.weight.data = torch.zeros(max_len, d_model)
        embedding_code.pos_emb.weight.requires_grad = False
    else:
        raise ValueError(f'PE {pe} not supported')
    code_encoder = TransformerEncoder(embedding_code, nhead=cfg["model_params"]["n_head_code"],
                                      dim_feedforward=cfg["model_params"]["dim_feedforward_code"],
                                      n_layers=cfg["model_params"]["n_layers_code"])
    # TODO custom model for nl
    embedding_nl = PositionEmbedding(vocab_size_nl, max_len=cfg["training_params"]["block_size_nl"],
                                     dim=cfg["model_params"]["d_model_nl"])
    nl_encoder = TransformerEncoder(embedding_nl, nhead=cfg["model_params"]["n_head_nl"],
                                    dim_feedforward=cfg["model_params"]["dim_feedforward_nl"],
                                    n_layers=cfg["model_params"]["n_layers_nl"])
    model = DualEconderModel(code_encoder, nl_encoder)
    return model


def build_model_checkpoint(vocab_size_code, vocab_size_nl, cfg):
    model = build_model(vocab_size_code, vocab_size_nl, cfg)
    model.load_state_dict(torch.load(cfg["checkpoint"]))
    return model
