import torch
import torch.nn as nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPo(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class MLPo_w_midfeature(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.out =  nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        y = self.out(x)
        return x, y


class MLP_L1(nn.Module):
    def __init__(self, dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    
class PreNorm2P(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, kv, **kwargs):
        return self.fn(self.norm1(q), self.norm2(kv), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, keyvalue):
        q = self.to_q(query)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = self.to_k(keyvalue)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = self.to_v(keyvalue)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
   

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CrossTransformer(nn.Module):
    def __init__(self, dim_kv, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., feature_size=96, patch_size=2):
        super().__init__()
        self.num_patches = feature_size//(patch_size*patch_size*patch_size)
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dim = dim
        self.patch_dim = dim_kv*patch_size*patch_size
        self.to_patch_embedding = nn.Linear(dim_kv*patch_size*patch_size, dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2P(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, q, kv):
        """
        q: query with shape of (b N d)
        kv: key and value with shape of (b N d) or (b c z h w)
        """
        if len(kv.shape) == 5:
            kv = rearrange(kv, 'b c (z p1) (h p2) (w p3) -> b (z h w) (c p1 p2 p3)', p1=self.patch_size, p2=self.patch_size, p3=self.patch_size)
        kv = self.to_patch_embedding(kv)
        kv = kv +  self.pos_embedding
        for attn, ff in self.layers:
            q = attn(q, kv) + q
            q = ff(q) + q
        return q


class StatusGeneration(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., hidden_dim=4, patch_size=4, image_size=48*96*96):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, image_size//8//(patch_size*patch_size*patch_size), dim))
        self.to_patch_embedding = nn.Linear(hidden_dim*2*patch_size*patch_size*patch_size, dim)
        self.feature_exactor= nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim*2),
            nn.ReLU()
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2P(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, q, mask):
        """
        q: query with shape of (b 1 d)
        mask: (b 1 z h w)
        """
        fmask =  self.feature_exactor(mask)
        kv = rearrange(fmask, 'b c (z p1) (h p2) (w p3) -> b (z h w) (c p1 p2 p3)', p1=self.patch_size, p2=self.patch_size, p3=self.patch_size)  
        kv = self.to_patch_embedding(kv)
        kv = kv +  self.pos_embedding
        for attn, ff in self.layers:
            q = attn(q, kv) + q
            q = ff(q) + q
        return q


class NodulePredictor(nn.Module):
    def __init__(self, factor_dim, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.to_patch_embedding = nn.Linear(factor_dim, dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm2P(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.to_class = nn.Linear(dim, 1)
    def forward(self, nodules, factors):
        """
        nodules: nodules with shape of (b 6 d)
        factors: factors with shape pf (b 5+1+10+6 d)
        """
        factors =self.to_patch_embedding(factors)
        for sa, ff1, ca, ff2 in self.layers:
            factors = sa(factors) + factors
            factors = ff1(factors) + factors
            nodules = ca(nodules, factors) + nodules
            nodules = ff2(nodules) + nodules
        nodules = self.to_class(nodules)
        return nodules