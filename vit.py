import torch.nn as nn
import torch
import torch.nn.functional as F

from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim=1024,
        num_layers=2,
        num_heads=16,
        qkv_bias=False,
        mlp_ratio=2.0,
        use_revised_ffn=False,
        dropout_rate=0.95,
        attn_dropout_rate=0.95,
        cls_head=False,
        num_classes=1,
        representation_size=None,
    ):
        super(VisionTransformer, self).__init__()

        # transformer
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)

        # output layer
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
            representation_size=representation_size,
            cls_head=cls_head,
        )

    def forward(self, x):
        x = self.transformer(x.view((-1, 1, 1024)))
        x = self.post_transformer_ln(x)
        x = self.cls_layer(x)
        return x
