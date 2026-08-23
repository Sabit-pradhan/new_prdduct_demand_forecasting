import torch
from torch import nn
class DemandNetwork(nn.Module):
 def __init__(self,ed,nd,bc,cc,c):
  super().__init__();self.brand_embedding=nn.Embedding(bc,c.brand_embedding_dim);self.category_embedding=nn.Embedding(cc,c.category_embedding_dim);n=ed+nd+c.brand_embedding_dim+c.category_embedding_dim;l=[]
  for h in c.hidden_dims:l += [nn.Linear(n,h),nn.ReLU(),nn.Dropout(c.dropout)];n=h
  self.backbone=nn.Sequential(*l);self.output=nn.Linear(n,3)
 def forward(self,e,n,b,c):return self.output(self.backbone(torch.cat([e,n,self.brand_embedding(b),self.category_embedding(c)],1)))