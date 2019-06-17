import torch
import math
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class Attention(nn.Module):
        def __init__(self, context_dim, hidden_dim, type="mlp"):
            super(Attention,self).__init__()
            self.context_dim = context_dim
            self.hidden_dim = hidden_dim
            self.type = type
            if type== "mlp":
                self.attn = nn.Linear(self.hidden_dim + self.context_dim, hidden_dim)
                self.v = nn.Parameter(torch.rand(hidden_dim))
            elif type == "last":
                None
            elif type == "mean":
                None
            elif type == "general" or type == "dot":
                raise NotImplementedError("General and Dot not implemented")
            else:
                raise  Exception("Wrong Atten Type")
            self.init_weight()

        def __repr__(self):
            s = "type = {}, context_dim= {}, hidden_dim= {}".format(self.type, self.context_dim, self.hidden_dim)
            return s
        def init_weight(self):
            if self.type == "mlp":
                nn.init.xavier_normal_(self.attn.weight)
                nn.init.uniform_(self.attn.bias,-0.1,0.1)
                stdv = 1. / math.sqrt(self.v.size(0))
                self.v.data.normal_(mean=0, std=stdv)
            elif self.type == "general":
                raise NotImplementedError("General and Dot not implemented")
            else:
                None
        def score(self, hidden, context):
            attn_input= torch.cat([hidden,context],dim=2)
            energy = F.tanh(self.attn(attn_input))  # [B*T*2H]->[B*T*H]
            energy = energy.transpose(2, 1)  # [B*H*T]
            v = self.v.repeat(context.size(0), 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]
            return energy.squeeze(1)  # [B*T]

        def forward(self, hidden, context):
            src_seq_len = context.size(0)

            if self.type == 'general' or self.type == 'dot':
                raise NotImplementedError("General and Dot not implemented")
            elif self.type == "last":
                attn_score = 1
                attn_context = torch.stack([context[0],context[-1]],dim=0)
                attn_context = torch.mean(attn_context,dim=0)
            elif self.type == "mean":
                attn_score = 1
                attn_context = torch.mean(context,dim=0)
            elif self.type == "mlp":
                H = hidden.unsqueeze(0).repeat(src_seq_len, 1, 1).transpose(0, 1)  # [B*T*H]
                context = context.transpose(0, 1)  # [B*T*H]
                attn_energies = self.score(H, context)  # compute attention score
                attn_score = nn.functional.softmax(attn_energies, dim=1)  # normalize with softmax [B*1*T]
                attn_context = attn_score.unsqueeze(1).bmm(context)
                attn_context = attn_context.squeeze(1)  # B*H

            return attn_score,attn_context
