"""Modular encoding layers for multimodal depression detection.

Includes:
- TransformerEncoderBlock: stacked self-attention + FFN
- ConvLSTMTemporalBlock: CNN + BiLSTM for temporal sequences (EEG, wearables)
- TextEncoderWithBackbone: BERT/RoBERTa backbone + CNN + BiLSTM + Transformer
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class TransformerEncoderBlock(nn.Module):
    """Stacked transformer encoder with self-attention and position-wise FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            key_padding_mask: [batch, seq_len] boolean mask (True = padding)
        Returns:
            [batch, seq_len, d_model]
        """
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)
        return out


class ConvLSTMTemporalBlock(nn.Module):
    """CNN + BiLSTM encoder for temporal sequences (EEG, wearables, MFCC, etc.)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_conv_layers: int = 2,
        conv_kernels: Tuple[int, ...] = (3, 5),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        conv_layers = []
        in_channels = input_dim
        for i, kernel_size in enumerate(conv_kernels[: min(num_conv_layers, len(conv_kernels))]):
            out_channels = hidden_dim if i == 0 else hidden_dim
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*conv_layers)
        self.bilstm = nn.LSTM(
            hidden_dim,
            max(hidden_dim // 2, 1),
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_proj = nn.Linear(max(hidden_dim // 2, 1) * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        if x.dim() == 3 and x.shape[-1] < x.shape[1]:
            x = x.transpose(1, 2)

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.shape[1] != self.cnn[0].in_channels:
            x = x.transpose(1, 2)

        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.bilstm(x)
        x = self.bilstm_proj(x)
        x = self.norm(x)
        return x


class TextEncoderWithBackbone(nn.Module):
    """Text encoder: BERT/RoBERTa backbone -> CNN -> BiLSTM -> Transformer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        backbone_name: Optional[str] = "bert-base-uncased",
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone_name

        if backbone_name is not None:
            try:
                from transformers import AutoModel
            except Exception as exc:
                raise RuntimeError(
                    "transformers is not installed. Install: pip install transformers torch"
                ) from exc

            self.backbone = AutoModel.from_pretrained(backbone_name)
            backbone_dim = int(self.backbone.config.hidden_size)
            self.input_adapter = nn.Linear(backbone_dim, hidden_dim)
        else:
            self.backbone = None
            self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.input_adapter = nn.Linear(embed_dim, hidden_dim)

        self.convlstm = ConvLSTMTemporalBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=2,
            dropout=dropout,
        )

        self.transformer = TransformerEncoderBlock(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=transformer_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1 = valid, 0 = padding)
        Returns:
            [batch, hidden_dim] pooled representation
        """
        if self.backbone is not None:
            backbone_out = self.backbone(input_ids=token_ids, attention_mask=attention_mask)
            x = backbone_out.last_hidden_state
        else:
            x = self.token_embedding(token_ids)

        x = self.input_adapter(x)
        x = self.convlstm(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        x = self.transformer(x, key_padding_mask=key_padding_mask)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)

        return pooled


class TemporalEncoderBlock(nn.Module):
    """Generic temporal encoder: CNN + BiLSTM + optional Transformer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_transformer: bool = False,
        transformer_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_transformer = use_transformer

        self.convlstm = ConvLSTMTemporalBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=2,
            dropout=dropout,
        )

        if use_transformer:
            self.transformer = TransformerEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                num_layers=transformer_layers,
                dropout=dropout,
            )
        else:
            self.transformer = None

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim, seq_len]
        Returns:
            [batch, hidden_dim]
        """
        x = self.convlstm(x)

        if self.use_transformer:
            x = self.transformer(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x
