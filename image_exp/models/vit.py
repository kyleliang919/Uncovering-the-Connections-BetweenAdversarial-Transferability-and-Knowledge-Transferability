import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ['vit']

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim=None, dropout_rate=0.0):
        super(MlpBlock,self).__init__()
        if out_dim is None:
            actual_out_dim = in_dim
        else:
            actual_out_dim = out_dim
        self.dense1 = nn.Linear(in_dim, mlp_dim)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.dense2 = nn.Linear(mlp_dim, actual_out_dim)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.gelu = nn.GELU()
        
    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return x
    
class Encoder1DBlock(nn.Module):
    def __init__(self, idx, seq_len, mlp_dim,hidden_size,dropout_rate = 0.0, attention_dropout_rate = 0.0, **attention_kwargs):
        super(Encoder1DBlock, self).__init__()
        num_heads = attention_kwargs['num_heads']
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=attention_dropout_rate)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlpblock = MlpBlock(in_dim = hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate)

    def forward(self, inputs):
        x = self.ln1(inputs)
        x = x.permute(1,0,2)
        x, _ = self.attn(x,x,x)
        x = x.permute(1,0,2)
        x = self.drop1(x)
        x = x + inputs
        y = self.ln2(x)
        y = self.mlpblock(y)
        return x + y
    
class Encoder(nn.Module):
    def __init__(self, seq_len, num_layers, mlp_dim, hidden_size, dropout_rate=0.0, **attention_kwargs):
        super(Encoder, self).__init__()
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList([
            Encoder1DBlock(
                idx=idx,
                seq_len=seq_len,
                mlp_dim = mlp_dim,
                hidden_size = hidden_size,
                dropout_rate = dropout_rate,
                **attention_kwargs
            ) for idx in range(num_layers)
        ])
        self.ln1 = nn.LayerNorm([hidden_size])
    
    def forward(self,inputs):
        x = inputs
        x = self.drop1(x)
        for lyr in self.layers:
            x = lyr(x)
        encoded = self.ln1(x)
        return encoded
    
class AddPositionEmbs(nn.Module):
    def __init__(self,seq_len, hidden_size):
        super(AddPositionEmbs, self).__init__()
        self.pe = Parameter(torch.randn(1,seq_len,hidden_size))
    def forward(self, inputs):
        return inputs + self.pe

class Embedding(nn.Module):
    def __init__(self, inputs, patches, hidden_size, transformer, num_classes=10, classifier='gap'):
        super(Embedding, self).__init__()
        c,h,w = inputs
        fh,fw = patches
        gh,gw = h//fh, w //fw
        self.classifier = classifier
        seq_len = gh * gw if classifier!= 'token' else gh * gw + 1
        self.conv = nn.Conv2d(c, hidden_size, kernel_size =(fh, fw), stride=(fh,fw))
        if self.classifier == 'token':
            self.cls = Parameter(torch.zeros(1,1,hidden_size))
        self.add_pos_embed = AddPositionEmbs(seq_len, hidden_size)
        
    def forward(self, x):
        x = self.conv(x)
        n,c,h,w = x.shape
        x = torch.reshape(x, [n, c, h*w])
        x = x.permute(0,2,1)
        if self.classifier == 'token':
            cls = self.cls.expand(n,-1,-1)
            x = torch.cat([cls, x], dim = 1)
        x = self.add_pos_embed(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, inputs, patches, num_layers, hidden_size, transformer, num_classes=10, classifier = 'gap'):
        super(VisionTransformer, self).__init__()
        c,h,w = inputs
        fh,fw = patches
        gh,gw = h//fh, w //fw
        seq_len = gh * gw if classifier != 'token' else gh * gw + 1
        self.encoder = Encoder(seq_len, num_layers, **transformer)
        self.dense_final = nn.Linear(hidden_size, num_classes)
        
    def forward(self,x):
        x = self.encoder(x)
        x = x[:,0,:]
        x = self.dense_final(x)
        return x

class VIT(nn.Module):
    def __init__(self, num_layers, num_classes, config):
        super(VIT, self).__init__()
        self.embed = Embedding(**config)
        self.transformer = VisionTransformer(num_layers = num_classes, num_classes = num_classes, **config)
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return x
        
def get_small_config():
    config = {}
    config['patches'] = (8,8)
    config['hidden_size'] = 768
    config['inputs'] = (3,32,32)
    config['transformer'] = {}
    config['transformer']['hidden_size'] = 768
    config['transformer']['mlp_dim'] = 3072
    config['transformer']['num_heads'] = 12
    config['transformer']['attention_dropout_rate'] = 0.0
    config['transformer']['dropout_rate'] = 0.1
    config['classifier'] = 'token'
    return config
    
def vit(**kwargs):
    model = VIT(12, 10, get_small_config())
    return model
