"""
IMUTransformer model
"""

import torch
from torch import nn
from torch.nn import Transformer

class IMUTransformer(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")
        self.input_proj = nn.Conv1d(6, transformer_dim, 1)
        self.window_size = config.get("window_size")

        self.transformer = Transformer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       num_encoder_layers = config.get("num_encoder_layers"),
                                       num_decoder_layers  = config.get("num_decoder_layers"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))




        self.query_embed = nn.Embedding(window_size, decoder_dim)
        config["output_dim"] = 6
        self.mlp_head =  MLPHead(config)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, data):
            src = data.get('imu')
            # Transformer pass
            target = self.transformer(self.input_proj(src), self.query_embed.weight).permute(1,2,0)
            target = self.mlp_head(target)
            return target

class MLPHead(nn.module):

    def __init__(self, config):
        mlp_activation = config.get("mlp_activation")
        output_dim = config.get("output_dim")
        self.mlp = nn.Sequential([nn.Conv1d(decoder_dim, decoder_dim/2, 1),
                                       self._get_activation(mlp_activation),
                                       nn.Conv1d(decoder_dim/2, decoder_dim /4, 1),
                                       self._get_activation(mlp_activation),
                                       nn.Conv1d(decoder_dim/4, output_dim, 1)])
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.mlp(x)

    def _get_activation(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return nn.Relu(inplace=True)
        if activation == "gelu":
            return nn.GELU()
        raise RuntimeError("Activation {} not supported".format(activation))
