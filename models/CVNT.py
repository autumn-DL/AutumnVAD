import torch
import torch.nn as nn
import torch.nn.functional as F

from module_lib.attention.base_attention import Attention
from module_lib.conv.base_conv import conform_conv

class conform_ffn(nn.Module):
    def __init__(self, dim, DropoutL1: float = 0.1, DropoutL2: float = 0.1):
        super().__init__()
        self.ln1 = nn.Linear(dim, dim * 4)
        self.ln2 = nn.Linear(dim * 4, dim)
        self.drop1 = nn.Dropout(DropoutL1) if DropoutL1 > 0. else nn.Identity()
        self.drop2 = nn.Dropout(DropoutL2) if DropoutL2 > 0. else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ln2(x)
        return self.drop2(x)
class conform_blocke(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, ffn_latent_drop: float = 0.1,
                 ffn_out_drop: float = 0.1, ):
        super().__init__()
        # self.ffn1 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.ffn2 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)


        self.conv = conform_conv(dim, kernel_size=kernel_size,

                                 DropoutL=conv_drop, )
        # self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)


    def forward(self, x, mask=None, ):
        # x = self.ffn1(self.norm1(x)) * 0.5 + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)

        x = self.conv(self.norm3(x)) + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.ffn2(self.norm4(x)) + x
        return x
class CVNT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_arg = config['model_arg']
        self.inlinear = nn.Linear(config['audio_num_mel_bins'], model_arg['encoder_conform_dim'])
        self.enc=nn.ModuleList([conform_blocke(
                                      dim=model_arg['encoder_conform_dim'],
                                      kernel_size=model_arg['encoder_conform_kernel_size'],
                                      ffn_latent_drop=model_arg['encoder_conform_ffn_latent_drop'],
                                      ffn_out_drop=model_arg['encoder_conform_ffn_out_drop'],

                                      ) for _ in range(model_arg['num_layers'])])
        self.outlinear = nn.Linear(model_arg['encoder_conform_dim'], 2)

    def forward(self, x, mask=None, ):
        # x=torch.transpose(x,1,2)

        x=self.inlinear(x)
        for i in self.enc:
            x = i(x, mask=mask)
        x=self.outlinear(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = torch.transpose(x, 1, 2)

        return x
