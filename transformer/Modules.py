import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

def isnan(inputs, name):
    if torch.any(torch.isnan(inputs)):
        print('isnan!!!!!!!in ',name)
        return True

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, st_weight=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        if st_weight is not None:
            attn = attn * st_weight

        attn = attn / self.temperature

        #print('attn',attn.size())

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn

class Origin_Interaction(nn.Module):
    ''' Interaction '''

    def __init__(self, temperature, attn_dropout=0.1, clip_size=3):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(clip_size * clip_size, 1)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, mask=None, st_weight=None, clip_size=3):

        sz_b, len_q, dk = q.size()
        sz_b, len_k, dk = k.size()
        # attn = torch.bmm(q, k.transpose(1, 2))
        q = q.unsqueeze_(2).expand(-1, -1, len_k, -1).contiguous()
        k = k.unsqueeze_(1).expand(-1, len_q, -1, -1).contiguous()
        itr_tensor = q * k   # b x lq x lk x dk
        global_weight = itr_tensor.sum(-1)  # b x lq x lk
        l1 = len_q // clip_size
        l2 = len_k // clip_size
        # step one
        itr_tensor = itr_tensor.view(sz_b,l1,clip_size,l2,clip_size,dk).permute(0,1,3,5,2,4).contiguous()
        itr_tensor = itr_tensor.view(sz_b,l1,l2,dk,-1)
        itr_tensor = self.fc(itr_tensor)
        # step two
        itr_tensor = itr_tensor.expand(-1,-1,-1,-1,clip_size*clip_size).contiguous()
        itr_tensor = itr_tensor.view(sz_b,l1,l2,dk,clip_size,clip_size).permute(0,1,4,2,5,3).contiguous()
        itr_tensor = itr_tensor.view(sz_b,l1,clip_size,-1,dk).view(sz_b,-1,len_k,dk)
        local_weight = itr_tensor.sum(-1)   # b x lq x lk
        attn = global_weight + local_weight
        if st_weight is not None:
            attn = attn * st_weight

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn

class Interaction(nn.Module):
    ''' Multi_Interaction  '''

    def __init__(self, dk, temperature, attn_dropout=0.1, clip_size_q=None, clip_size_k=None):
        super().__init__()

        print('[Checking here] clip_size_q {}; clip_size_k {}'.format(clip_size_q, clip_size_k))
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(clip_size_q * clip_size_k, 1)
        self.downsample = nn.Conv2d(dk, dk, (clip_size_q,clip_size_k), stride=(clip_size_q,clip_size_k))
        self.upsample = nn.ConvTranspose2d(dk, dk, (clip_size_q,clip_size_k), stride=(clip_size_q,clip_size_k))
        self.clip_size_q = clip_size_q
        self.clip_size_k = clip_size_k
        self.relu = torch.nn.ReLU()

        nn.init.xavier_normal_(self.fc.weight)


    def forward(self, q, k, v, mask=None, st_weight=None, use_conv=True):

        sz_b, len_q, dk = q.size()
        sz_b, len_k, dk = k.size()
        clip_size_q = self.clip_size_q
        clip_size_k = self.clip_size_k
        # attn = torch.bmm(q, k.transpose(1, 2))
        q = q.unsqueeze_(2).expand(-1, -1, len_k, -1).contiguous()
        k = k.unsqueeze_(1).expand(-1, len_q, -1, -1).contiguous()
        itr_tensor = q * k   # b x lq x lk x dk
        global_weight = itr_tensor.sum(-1)  # b x lq x lk
        #print('len_q',len_q,'len_k',len_k,'clip_size_q',clip_size_q,'clip_size_k',clip_size_k)
        l1 = len_q // clip_size_q
        l2 = len_k // clip_size_k
        if not use_conv:
            # step one
            itr_tensor = itr_tensor.view(sz_b,l1,clip_size_q,l2,clip_size_k,dk).permute(0,1,3,5,2,4).contiguous()
            itr_tensor = itr_tensor.view(sz_b,l1,l2,dk,-1)
            itr_tensor = self.relu(self.fc(itr_tensor))
            # step two
            itr_tensor = itr_tensor.expand(-1,-1,-1,-1,clip_size_q*clip_size_k).contiguous()
            itr_tensor = itr_tensor.view(sz_b,l1,l2,dk,clip_size_q,clip_size_k).permute(0,1,4,2,5,3).contiguous()
            itr_tensor = itr_tensor.view(sz_b,l1,clip_size_q,-1,dk).view(sz_b,-1,len_k,dk)
        else:
            # b*lq*lk*dk --> b*dk*lq*lk
            itr_tensor = itr_tensor.permute(0,3,1,2).contiguous()
            target_size = itr_tensor.size()
            itr_tensor = self.downsample(itr_tensor)
            itr_tensor = self.upsample(itr_tensor,target_size)
            # b*dk*lq*lk --> b*lq*lk*dk
            itr_tensor = itr_tensor.permute(0,2,3,1).contiguous()

        local_weight = itr_tensor.sum(-1)   # b x lq x lk
        attn = 0.5*global_weight + 0.5*local_weight
        if st_weight is not None:
            attn = attn * st_weight

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn

