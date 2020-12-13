"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")
        self.input_proj = nn.Conv1d(6, self.transformer_dim, 1)
        self.window_size = config.get("window_size")

        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm=nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        config["output_dim"] = config.get("num_classes")
        self.imu_head = IMUHead(config)

        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]

        # Class probability
        target = self.log_softmax(self.imu_head(target))
        return target


class IMUHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        mlp_activation = config.get("head_activation")
        output_dim = config.get("output_dim")
        encoder_dim = config.get("transformer_dim")
        self.mlp = nn.Sequential(nn.Linear(encoder_dim, encoder_dim // 2),
                                       get_activation(mlp_activation),
                                       nn.Linear(encoder_dim // 2, encoder_dim // 4),
                                       get_activation(mlp_activation),
                                       nn.Linear(encoder_dim // 4, output_dim))
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.mlp(x)


def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))
