import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, RobertaModel


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
        return emb_code['pooler_output'], emb_nl['pooler_output']


def build_model(tokenizer_code, tokenizer_nl, cfg):
    roberta_config_nl = AutoConfig.from_pretrained('roberta-base')
    roberta_config_nl.vocab_size = len(tokenizer_nl)
    roberta_config_nl.max_position_embeddings = cfg["training_params"]["block_size_nl"] + 2
    roberta_config_nl.pad_token_id = tokenizer_nl.convert_tokens_to_ids(tokenizer_nl.pad_token)
    roberta_config_nl.sep_token_id = tokenizer_nl.convert_tokens_to_ids(tokenizer_nl.sep_token)
    roberta_config_nl.eos_token_id = tokenizer_nl.convert_tokens_to_ids(tokenizer_nl.eos_token)
    roberta_config_nl.num_hidden_layers = cfg["model_params"]["n_layers"]
    nl_encoder = RobertaModel(roberta_config_nl)

    roberta_config_code = AutoConfig.from_pretrained('roberta-base')
    roberta_config_code.vocab_size = len(tokenizer_code)
    roberta_config_code.max_position_embeddings = cfg["training_params"]["block_size_code"] + 2
    roberta_config_code.pad_token_id = tokenizer_code.convert_tokens_to_ids(tokenizer_code.pad_token)
    roberta_config_code.sep_token_id = tokenizer_code.convert_tokens_to_ids(tokenizer_code.sep_token)
    roberta_config_code.eos_token_id = tokenizer_code.convert_tokens_to_ids(tokenizer_code.eos_token)
    roberta_config_code.num_hidden_layers = cfg["model_params"]["n_layers"]
    code_encoder = RobertaModel(roberta_config_code)

    if cfg["pe_model"] == "random":
        pass
    elif cfg["pe_model"] == "sinusoid":
        sinusoid_pe = get_sinusoid(cfg["training_params"]["block_size_code"], roberta_config_code.hidden_size)
        pad_id = tokenizer_code.convert_tokens_to_ids(tokenizer_code.pad_token)
        code_encoder.embeddings.position_embeddings.weight.data[pad_id + 1:, :] = sinusoid_pe
        code_encoder.embeddings.position_embeddings.weight.requires_grad = False
    elif cfg["pe_model"] == "gpt2":
        gpt2_pe = AutoModel.from_pretrained(cfg["pe_model"]).wpe.weight.data[0:cfg["training_params"]["block_size_code"]]
        pad_id = tokenizer_code.convert_tokens_to_ids(tokenizer_code.pad_token)
        code_encoder.embeddings.position_embeddings.weight.data[pad_id + 1:, :] = gpt2_pe
        code_encoder.embeddings.position_embeddings.weight.requires_grad = False
    elif cfg["pe_model"] == "roberta-base":
        roberta_pe = AutoModel.from_pretrained(cfg["pe_model"]).embeddings.position_embeddings.weight.data[
                  2:cfg["training_params"]["block_size_code"] + 2]
        pad_id = tokenizer_code.convert_tokens_to_ids(tokenizer_code.pad_token)
        code_encoder.embeddings.position_embeddings.weight.data[pad_id + 1:, :] = roberta_pe
        code_encoder.embeddings.position_embeddings.weight.requires_grad = False
    model = DualEconderModel(code_encoder, nl_encoder)
    return model


def build_model_checkpoint(tokenizer_code, tokenizer_nl, cfg):
    model = build_model(tokenizer_code, tokenizer_nl, cfg)
    model.load_state_dict(torch.load(cfg["checkpoint"]))
    return model
