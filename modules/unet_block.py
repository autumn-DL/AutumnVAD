import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class RCB(nn.Module):
    def __init__(self, indim, outdim, K=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=indim, out_channels=outdim, kernel_size=K, padding=K // 2)
        self.ppj = nn.Conv2d(in_channels=indim, out_channels=outdim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=outdim, out_channels=outdim, kernel_size=K, padding=K // 2)
        self.bn1 = nn.BatchNorm2d(outdim)
        self.bn2 = nn.BatchNorm2d(outdim)
        self.act = nn.ReLU()

    def forward(self, x):
        '''

        :param x: B C H W
        :return:
        '''

        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))) + self.ppj(x)


class REB(nn.Module):
    def __init__(self, indim, outdim, nblock=4, K=3):
        super().__init__()
        self.RCB = nn.ModuleList()
        self.RCB.append(RCB(indim, outdim, K=K))
        for _ in range(nblock - 1):
            self.RCB.append(RCB(outdim, outdim, K=K))
        self.avgp = nn.AvgPool2d(2, 2)

    def forward(self, x):

        for i in self.RCB:
            x = i(x)
        return x, self.avgp(x)


class IEB(nn.Module):
    def __init__(self, indim, outdim, nblock=4, K=3):
        super().__init__()
        self.RCB = nn.ModuleList()
        self.RCB.append(RCB(indim, outdim, K=K))
        for _ in range(nblock - 1):
            self.RCB.append(RCB(outdim, outdim, K=K))

    def forward(self, x):

        for i in self.RCB:
            x = i(x)
        return x


class RDB(nn.Module):
    def __init__(self, indim, outdim, nblock=4, K=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(indim, outdim, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(outdim)
        self.act = nn.ReLU()
        self.RCB = nn.ModuleList()
        self.RCBRES = (RCB(outdim * 2, outdim, K=K))
        for _ in range(nblock - 1):
            self.RCB.append(RCB(outdim, outdim, K=K))

    def forward(self, x, res):
        x = self.act(self.bn(self.up(x)))
        x = self.RCBRES(torch.cat([x, res], dim=1))
        for i in self.RCB:
            x = i(x)
        return x


class RMVU(nn.Module):
    def __init__(self,deep=5,indim=1,blockdim=[16,32,64,128,256],middeldim=512,nblocks=4,K=3):
        super().__init__()
        self.down=nn.ModuleList()
        self.up=nn.ModuleList()
        self.middel=IEB(blockdim[-1], outdim=middeldim, K=K,nblock=nblocks)
        self.bn=nn.BatchNorm2d(indim)
        for i in range(deep):
            if i==0:
                self.down.append(REB(indim,blockdim[i],nblock=nblocks,K=K))
            else:
                self.down.append(REB( blockdim[i-1], blockdim[i], nblock=nblocks, K=K))
        blockdim.reverse()
        for i in range(deep):
            if i==0:
                self.up.append(RDB(middeldim,blockdim[i],nblock=nblocks,K=K))
            else:
                self.up.append(RDB( blockdim[i-1], blockdim[i], nblock=nblocks, K=K))

    def forward(self,x):

        res=[]
        for i in self.down:
            res1,x=i(x)
            res.append(res1)
        x=self.middel(x)
        res.reverse()
        for i ,res2 in zip(self.up,res):
            x=i(x,res2)

        return x
if __name__=='__main__':
    from einops import rearrange
    ub=RMVU()
    st=ub(torch.randn(1,1,96,320))
    B,C,H,W=st.size()
    st2=rearrange(st, "b c h w -> b (c h) w",)
    pass

