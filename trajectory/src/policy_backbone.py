import torch
import torch.nn as nn
from .depth_anything.depth_anything_v2.dpt import DepthAnythingV2

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:x.size(1)]

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
        
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.eval()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256) 
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images,depths):
        with torch.no_grad():
            if len(images.shape) == 4:
                tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            elif len(images.shape) == 5:
                tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
            if len(depths.shape) == 4:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif len(depths.shape) == 5:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
            former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384),device=self.device))
            memory_token = self.former_net(former_query,former_token)
            memory_token = self.project_layer(memory_token)
            return memory_token